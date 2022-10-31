// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MAP_H_
#define CARBON_COMMON_MAP_H_

#include <x86intrin.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ReverseIteration.h"
#include "llvm/Support/type_traits.h"

namespace Carbon {

template <typename KeyT, typename ValueT> class MapView;
template <typename KeyT, typename ValueT> class MapRef;
template <typename KeyT, typename ValueT, int MinSmallSize> class Map;

namespace map_internal {

// Detect whether we can use SIMD accelerated implementations of the control
// groups.
#if defined(__SSSE3__)
#define CARBON_USE_SSE_CONTROL_GROUP 1
#if defined(__SSE4_1__)
#define CARBON_OPTIMIZE_SSE4_1 1
#endif
#endif

struct MapInfo {
  int Size;
  int GrowthBudget;
  int Entropy;
};

template <typename KeyT, typename ValueT> class LookupKVResult {
public:
  LookupKVResult() = default;
  LookupKVResult(KeyT *Key, ValueT *Value) : Key(Key), Value(Value) {}

  explicit operator bool() const { return Key != nullptr; }

  KeyT &getKey() const { return *Key; }
  ValueT &getValue() const { return *Value; }

private:
  KeyT *Key = nullptr;
  ValueT *Value = nullptr;
};

template <typename KeyT, typename ValueT> class InsertKVResult {
public:
  InsertKVResult() = default;
  InsertKVResult(bool Inserted, KeyT &Key, ValueT &Value)
      : KeyAndInserted(&Key, Inserted), Value(&Value) {}

  bool isInserted() const { return KeyAndInserted.getInt(); }

  KeyT &getKey() const { return *KeyAndInserted.getPointer(); }
  ValueT &getValue() const { return *Value; }

private:
  llvm::PointerIntPair<KeyT *, 1, bool> KeyAndInserted;
  ValueT *Value = nullptr;
};

// We organize the hashtable into 16-slot groups so that we can use 16 control
// bytes to understand the contents efficiently.
constexpr int GroupSize = 16;

template <typename KeyT, typename ValueT> struct Group;
template <typename KeyT, typename ValueT, bool IsCmpVec = true> class GroupMatchedByteRange;

template <typename KeyT, typename ValueT> class GroupMatchedByteIterator
    : public llvm::iterator_facade_base<
          GroupMatchedByteIterator<KeyT, ValueT>, std::forward_iterator_tag, int, int> {
  friend struct Group<KeyT, ValueT>;
  friend class GroupMatchedByteRange<KeyT, ValueT, /*IsCmpVec=*/true>;
  friend class GroupMatchedByteRange<KeyT, ValueT, /*IsCmpVec=*/false>;

  ssize_t ByteIndex;

  unsigned Mask = 0;

  GroupMatchedByteIterator(unsigned Mask) : Mask(Mask) {}

public:
  GroupMatchedByteIterator() = default;

  bool operator==(const GroupMatchedByteIterator &RHS) const {
    return Mask == RHS.Mask;
  }

  ssize_t &operator*() {
    assert(Mask != 0 && "Cannot get an index from a zero mask!");
    ByteIndex = llvm::countTrailingZeros(Mask, llvm::ZB_Undefined);
    return ByteIndex;
  }

  GroupMatchedByteIterator &operator++() {
    assert(Mask != 0 && "Must not be called with a zero mask!");
    Mask &= (Mask - 1);
    return *this;
  }
};

template <typename KeyT, typename ValueT, bool IsCmpVec> class GroupMatchedByteRange {
  friend struct Group<KeyT, ValueT>;
  using MatchedByteIterator = GroupMatchedByteIterator<KeyT, ValueT>;

#if CARBON_OPTIMIZE_SSE4_1
  __m128i MaskVec;

  GroupMatchedByteRange(__m128i MaskVec) : MaskVec(MaskVec) {}
#else
  unsigned Mask;

  GroupMatchedByteRange(unsigned Mask) : Mask(Mask) {}
#endif

public:
  GroupMatchedByteRange() = default;

  /// Returns false if this range is empty. Provided as a conversion to
  /// simplify usage with condition variables prior to C++17.
  ///
  /// Because testing for emptiness is potentially optimized, we want to
  /// encourage guarding with an empty test.
  explicit operator bool() const { return !empty(); }

  /// Potentially optimized test for empty.
  ///
  /// With appropriate SIMD support, this may be faster than beginning
  /// iteration. Even when it isn't optimized, any duplication with the
  /// initial test for a loop should get eliminated during optimization.
  bool empty() const {
#if CARBON_OPTIMIZE_SSE4_1
    return _mm_test_all_zeros(
        MaskVec, IsCmpVec ? MaskVec : _mm_set1_epi8(static_cast<char>(0b10000000u)));
#else
    return Mask == 0;
#endif
  }

  MatchedByteIterator begin() const {
#if CARBON_OPTIMIZE_SSE4_1
    unsigned Mask = _mm_movemask_epi8(MaskVec);
    return MatchedByteIterator(Mask);
#else
    return MatchedByteIterator(Mask);
#endif
  }

  MatchedByteIterator end() const { return MatchedByteIterator(); }
};

template <typename KeyT, typename ValueT>
struct Group {
  // Each control byte can have special values. All special values have the
  // most significant bit set to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  template <bool IsCmpVec = true>
  using MatchedByteRange = GroupMatchedByteRange<KeyT, ValueT, IsCmpVec>;

#if CARBON_USE_SSE_CONTROL_GROUP
  __m128i ByteVec = {};
#else
  alignas(GroupSize) std::array<uint8_t, GroupSize> Bytes = {};
#endif

  // Now we need storage for the keys in this group and the values in this
  // group. We put the keys first to pack as many keys onto the same cacheline
  // as the control bytes, and then onto linear subsequent cache lines to
  // facilitate linear prefetching pulling those cache lines in advance.
  union {
    KeyT Keys[GroupSize];
  };
  union {
    ValueT Values[GroupSize];
  };

  MatchedByteRange<> match(uint8_t MatchByte) const {
#if CARBON_USE_SSE_CONTROL_GROUP
    auto MatchByteVec = _mm_set1_epi8(MatchByte);
    auto MatchByteCmpVec = _mm_cmpeq_epi8(ByteVec, MatchByteVec);
#if CARBON_OPTIMIZE_SSE4_1
    return {MatchByteCmpVec};
#else
    return {(unsigned)_mm_movemask_epi8(MatchByteCmpVec)};
#endif
#else
    unsigned Mask = 0;
    for (int ByteIndex : llvm::seq(0, GroupSize))
      Mask |= (Bytes[ByteIndex] == MatchByte) << ByteIndex;
    return {Mask};
#endif
  }

  MatchedByteRange<> matchEmpty() const {
    return match(Empty);
  }

  MatchedByteRange<> matchDeleted() const {
    return match(Deleted);
  }

  MatchedByteRange</*IsCmpVec=*/false> matchPresent() const {
#if CARBON_USE_SSE_CONTROL_GROUP
#if CARBON_OPTIMIZE_SSE4_1
    return {ByteVec};
#else
    // We arrange the byte vector for present bytes so that we can directly
    // extract it as a mask.
    return {(unsigned)_mm_movemask_epi8(ByteVec)};
#endif
#else
    // Generic code to compute a bitmask.
    unsigned Mask = 0;
    for (int ByteIndex : llvm::seq(0, GroupSize))
      Mask |= (bool)(Bytes[ByteIndex] & 0b10000000u) << ByteIndex;
    return {Mask};
#endif
  }

  void setByte(int ByteIndex, uint8_t ControlByte) {
#if CARBON_USE_SSE_CONTROL_GROUP
    // We directly manipulate the storage of the vector as there isn't a nice
    // intrinsic for this.
    ((unsigned char *)&ByteVec)[ByteIndex] = ControlByte;
#else
    Bytes[ByteIndex] = ControlByte;
#endif
  }

  void clear() {
#if CARBON_USE_SSE_CONTROL_GROUP
    ByteVec = _mm_set1_epi8(Empty);
#else
    for (int ByteIndex : llvm::seq(0, GroupSize))
      Bytes[ByteIndex] = Empty;
#endif
  }
};

constexpr int EntropyMask = INT_MAX & ~(GroupSize - 1);

template <typename KeyT> constexpr int getKeysInCacheline() {
  constexpr int CachelineSize = 64;

  return CachelineSize / sizeof(KeyT);
}

template <typename KeyT> constexpr bool shouldUseLinearLookup(int SmallSize) {
  return SmallSize <= getKeysInCacheline<KeyT>();
}

template <typename KeyT, typename ValueT, bool UseLinearLookup, int SmallSize> struct SmallSizeBuffer;

template <typename KeyT, typename ValueT> struct SmallSizeBuffer<KeyT, ValueT, true, 0> {};

template <typename KeyT, typename ValueT, int SmallSize> struct SmallSizeBuffer<KeyT, ValueT, true, SmallSize> {
  union {
    KeyT Keys[SmallSize];
  };
  union {
    ValueT Values[SmallSize];
  };
};

template <typename KeyT, typename ValueT, int SmallSize> struct SmallSizeBuffer<KeyT, ValueT, false, SmallSize> {
  using GroupT = map_internal::Group<KeyT, ValueT>;

  // FIXME: One interesting question is whether the small size should be a minumum here or an exact figure.
  static_assert(llvm::isPowerOf2_32(SmallSize),
                "SmallSize must be a power of two for a hashed buffer!");
  static_assert(SmallSize >= map_internal::GroupSize,
                "SmallSize must be at least the size of one group!");
  static_assert((SmallSize % map_internal::GroupSize) == 0,
                "SmallSize must be a multiple of the group size!");
  static constexpr int SmallNumGroups = SmallSize / map_internal::GroupSize;
  static_assert(llvm::isPowerOf2_32(SmallNumGroups),
                "The number of groups must be a power of two when hashing!");

  GroupT Groups[SmallNumGroups];
};

template <typename KeyT, typename ValueT, typename LookupKeyT>
bool contains(const LookupKeyT &LookupKey, int Size, int Entropy, int SmallSize,
              void *Data);

template <typename KeyT, typename ValueT, typename LookupKeyT>
LookupKVResult<KeyT, ValueT> lookup(const LookupKeyT &LookupKey, int Size,
                                    int Entropy, int SmallSize, void *Data);

template <typename KeyT, typename ValueT, typename CallbackT>
void forEach(int Size, int Entropy, int SmallSize, void *Data, CallbackT Callback);

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT = Group<KeyT, ValueT>>
InsertKVResult<KeyT, ValueT>
insert(const LookupKeyT &LookupKey,
       llvm::function_ref<std::pair<KeyT *, ValueT *>(
           const LookupKeyT &LookupKey, void *KeyStorage, void *ValueStorage)>
           InsertCB,
       MapInfo &Info, int SmallSize, void *Data, GroupT *&GroupsPtr);

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT = Group<KeyT, ValueT>>
InsertKVResult<KeyT, ValueT>
update(const LookupKeyT &LookupKey,
       llvm::function_ref<std::pair<KeyT *, ValueT *>(
           const LookupKeyT &LookupKey, void *KeyStorage, void *ValueStorage)>
           InsertCB,
       llvm::function_ref<ValueT &(KeyT &Key, ValueT &Value)> UpdateCB, MapInfo &Info,
       int SmallSize, void *Data, GroupT *&GroupsPtr);

template <typename KeyT, typename ValueT, typename LookupKeyT>
bool erase(const LookupKeyT &LookupKey, MapInfo &Info, int SmallSize,
           void *Data);

template <typename KeyT, typename ValueT>
void clear(MapInfo &Info, int SmallSize, void *Data);

template <typename KeyT, typename ValueT>
void init(MapInfo &Info, int SmallSize, void *SmallBufferAddr, void *ThisAddr);

template <typename KeyT, typename ValueT, typename GroupT = Group<KeyT, ValueT>>
void reset(MapInfo &Info, int SmallSize, void *SmallBufferAddr, void *ThisAddr,
           GroupT *GroupsPtr);

template <typename KeyT, typename ValueT>
void copyInit(MapInfo &Info, int SmallSize, void *SmallBufferAddr, void *ThisAddr,
              const MapInfo &ArgInfo,  void *ArgSmallBufferAddr, void *ArgAddr);

} // namespace map_internal

template <typename InputKeyT, typename InputValueT> class MapView {
public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using LookupKVResultT = typename map_internal::LookupKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  bool contains(const LookupKeyT &LookupKey) const {
    return map_internal::contains<KeyT, ValueT>(LookupKey, Size, Entropy,
                                                SmallSize, Data);
  }

  template <typename LookupKeyT>
  LookupKVResultT lookup(const LookupKeyT &LookupKey) const {
    return map_internal::lookup<KeyT, ValueT>(LookupKey, Size, Entropy,
                                              SmallSize, Data);
  }

  template <typename LookupKeyT>
  ValueT *operator[](const LookupKeyT &LookupKey) const {
    auto Result = lookup(LookupKey);
    return Result ? &Result.getValue() : nullptr;
  }

  template <typename CallbackT>
  void forEach(CallbackT Callback) {
    map_internal::forEach<KeyT, ValueT>(Size, Entropy, SmallSize,
                                        Data, Callback);
  }

private:
  template <typename MapKeyT, typename MapValueT, int MinSmallSize>
  friend class Map;
  friend class MapRef<KeyT, ValueT>;

  MapView(int Size, int Entropy, int SmallSize, void *Data)
      : Size(Size), Entropy(Entropy), SmallSize(SmallSize), Data(Data) {}

  int Size;
  int Entropy;
  int SmallSize;
  void *Data;

  template <typename LookupKeyT>
  static bool contains(const LookupKeyT &LookupKey, int Size, int Entropy,
                       int SmallSize, void *Data);

  template <typename LookupKeyT>
  static LookupKVResultT lookup(const LookupKeyT &LookupKey, int Size,
                                int Entropy, int SmallSize, void *Data);
};

template <typename InputKeyT, typename InputValueT> class MapRef {
public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using LookupKVResultT = map_internal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = map_internal::InsertKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  bool contains(const LookupKeyT &LookupKey) const {
    return map_internal::contains<KeyT, ValueT>(LookupKey, getSize(), getEntropy(),
                                                SmallSize, getRawData());
  }

  template <typename LookupKeyT>
  LookupKVResultT lookup(const LookupKeyT &LookupKey) const {
    return map_internal::lookup<KeyT, ValueT>(LookupKey, getSize(), getEntropy(),
                                              SmallSize, getRawData());
  }

  template <typename LookupKeyT>
  ValueT *operator[](const LookupKeyT &LookupKey) const {
    auto Result = lookup(LookupKey);
    return Result ? &Result.getValue() : nullptr;
  }

  operator ViewT() const {
    return {getSize(), getEntropy(), SmallSize, getRawData()};
  }

  template <typename LookupKeyT>
  InsertKVResultT
  insert(const LookupKeyT &LookupKey,
         typename std::__type_identity<llvm::function_ref<std::pair<KeyT *, ValueT *>(
             const LookupKeyT &LookupKey, void *KeyStorage,
             void *ValueStorage)>>::type InsertCB) {
    return map_internal::insert<KeyT, ValueT, LookupKeyT>(
        LookupKey, InsertCB, *Info, SmallSize, getRawData(), getGroupsPtr());
  }

  template <typename LookupKeyT>
  InsertKVResultT insert(const LookupKeyT &LookupKey, ValueT NewV) {
    return insert(LookupKey,
                  [&NewV](const LookupKeyT &LookupKey, void *KeyStorage,
                          void *ValueStorage) -> std::pair<KeyT *, ValueT *> {
                    KeyT *K = new (KeyStorage) KeyT(LookupKey);
                    ValueT *V = new (ValueStorage) ValueT(std::move(NewV));
                    return {K, V};
                  });
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  typename std::enable_if<
      !std::is_same<ValueT, ValueCallbackT>::value &&
          std::is_same<ValueT,
                       decltype(std::declval<ValueCallbackT>()())>::value,

      InsertKVResultT>::type
  insert(const LookupKeyT &LookupKey, ValueCallbackT ValueCB) {
    return insert(
        LookupKey,
        [&ValueCB](const LookupKeyT &LookupKey, void *KeyStorage,
                   void *ValueStorage) -> std::pair<KeyT *, ValueT *> {
          KeyT *K = new (KeyStorage) KeyT(LookupKey);
          ValueT *V = new (ValueStorage) ValueT(ValueCB());
          return {K, V};
        });
  }

  template <typename LookupKeyT>
  InsertKVResultT
  update(const LookupKeyT &LookupKey,
         typename std::__type_identity<llvm::function_ref<std::pair<KeyT *, ValueT *>(
             const LookupKeyT &LookupKey, void *KeyStorage,
             void *ValueStorage)>>::type InsertCB,
         llvm::function_ref<ValueT &(KeyT &Key, ValueT &Value)> UpdateCB) {
    return map_internal::update<KeyT, ValueT>(LookupKey, InsertCB, UpdateCB,
                                              *Info, SmallSize, getRawData(),
                                              getGroupsPtr());
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  typename std::enable_if<
      !std::is_same<ValueT, ValueCallbackT>::value &&
          std::is_same<ValueT,
                       decltype(std::declval<ValueCallbackT>()())>::value,

      InsertKVResultT>::type
  update(const LookupKeyT &LookupKey, ValueCallbackT ValueCB) {
    return update(
        LookupKey,
        [&ValueCB](const LookupKeyT &LookupKey, void *KeyStorage,
                   void *ValueStorage) -> std::pair<KeyT *, ValueT *> {
          KeyT *K = new (KeyStorage) KeyT(LookupKey);
          ValueT *V = new (ValueStorage) ValueT(ValueCB());
          return {K, V};
        },
        [&ValueCB](KeyT &/*Key*/, ValueT &Value) -> ValueT & {
          Value.~ValueT();
          return *new (&Value) ValueT(ValueCB());
        });
  }

  template <typename LookupKeyT>
  InsertKVResultT update(const LookupKeyT &LookupKey, ValueT NewV) {
    return update(
        LookupKey,
        [&NewV](const LookupKeyT &LookupKey, void *KeyStorage,
                void *ValueStorage) -> std::pair<KeyT *, ValueT *> {
          KeyT *K = new (KeyStorage) KeyT(LookupKey);
          ValueT *V = new (ValueStorage) ValueT(std::move(NewV));
          return {K, V};
        },
        [&NewV](KeyT &/*Key*/, ValueT &Value) -> ValueT & {
          Value.~ValueT();
          return *new (&Value) ValueT(std::move(NewV));
        });
  }

  template <typename LookupKeyT> bool erase(const LookupKeyT &LookupKey) {
    return map_internal::erase<KeyT, ValueT>(LookupKey, *Info, SmallSize,
                                             getRawData());
  }

  template <typename CallbackT>
  void forEach(CallbackT Callback) {
    map_internal::forEach<KeyT, ValueT>(getSize(), getEntropy(), SmallSize,
                                        getRawData(), Callback);
  }

  void clear() {
    map_internal::clear<KeyT, ValueT>(*Info, SmallSize, getRawData());
  }

  void reset() {
    map_internal::reset<KeyT, ValueT>(*Info, SmallSize, getSmallBufferAddr(),
                                      Info, getGroupsPtr());
  }

private:
  template <typename MapKeyT, typename MapValueT, int MinSmallSize>
  friend class Map;

  using GroupT = map_internal::Group<KeyT, ValueT>;

  MapRef(map_internal::MapInfo &Info, int SmallSize, GroupT *&GroupsPtr)
      : Info(&Info), SmallSize(SmallSize), GroupsPtrOrSmallBuffer(&GroupsPtr) {}

  map_internal::MapInfo *Info;
  int SmallSize;
  GroupT **GroupsPtrOrSmallBuffer;

  int getSize() const { return Info->Size; }
  int getGrowthBudget() const { return Info->GrowthBudget; }
  int getEntropy() const { return Info->Entropy; }

  void *getSmallBufferAddr() const { return GroupsPtrOrSmallBuffer; }
  GroupT *&getGroupsPtr() const {
    return *reinterpret_cast<GroupT **>(GroupsPtrOrSmallBuffer);
  }

  bool isSmall() const { return getEntropy() < 0; }

  void *getRawData() const {
    return isSmall() ? getSmallBufferAddr() : getGroupsPtr();
  }
};

template <typename InputKeyT, typename InputValueT, int MinSmallSize = 0> class Map {
public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using RefT = MapRef<KeyT, ValueT>;
  using LookupKVResultT = map_internal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = map_internal::InsertKVResult<KeyT, ValueT>;

  Map() {
    map_internal::init<KeyT, ValueT>(Info, SmallSize, &SmallBuffer, &Info);
  }
  ~Map() {
    if (!isSmall()) {
      // Not using a small buffer, so we need to deallocate it.
      delete[] AllocatedGroups;
    }
  }
  Map(const Map &Arg) {
    map_internal::copyInit<KeyT, ValueT>(Info, SmallSize, &SmallBuffer, &Info,
                                         Arg.Info, &Arg.SmallBuffer, &Arg.Info);
  }
  Map(Map &&Arg) = delete;

  template <typename LookupKeyT>
  bool contains(const LookupKeyT &LookupKey) const {
    return map_internal::contains<KeyT, ValueT>(LookupKey, getSize(), getEntropy(),
                                                SmallSize, getRawData());
  }

  template <typename LookupKeyT>
  LookupKVResultT lookup(const LookupKeyT &LookupKey) const {
    return map_internal::lookup<KeyT, ValueT>(LookupKey, getSize(), getEntropy(),
                                              SmallSize, getRawData());
  }

  template <typename LookupKeyT>
  ValueT *operator[](const LookupKeyT &LookupKey) const {
    auto Result = lookup(LookupKey);
    return Result ? &Result.getValue() : nullptr;
  }

  operator ViewT() const {
    return {getSize(), getEntropy(), SmallSize, getRawData()};
  }

  template <typename LookupKeyT>
  InsertKVResultT
  insert(const LookupKeyT &LookupKey,
         typename std::__type_identity<llvm::function_ref<std::pair<KeyT *, ValueT *>(
             const LookupKeyT &LookupKey, void *KeyStorage,
             void *ValueStorage)>>::type InsertCB) {
    return map_internal::insert<KeyT, ValueT, LookupKeyT>(
        LookupKey, InsertCB, Info, SmallSize, getRawData(), getGroupsPtr());
  }

  template <typename LookupKeyT>
  InsertKVResultT insert(const LookupKeyT &LookupKey, ValueT NewV) {
    return insert(LookupKey,
                  [&NewV](const LookupKeyT &LookupKey, void *KeyStorage,
                          void *ValueStorage) -> std::pair<KeyT *, ValueT *> {
                    KeyT *K = new (KeyStorage) KeyT(LookupKey);
                    ValueT *V = new (ValueStorage) ValueT(std::move(NewV));
                    return {K, V};
                  });
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  typename std::enable_if<
      !std::is_same<ValueT, ValueCallbackT>::value &&
          std::is_same<ValueT,
                       decltype(std::declval<ValueCallbackT>()())>::value,

      InsertKVResultT>::type
  insert(const LookupKeyT &LookupKey, ValueCallbackT ValueCB) {
    return insert(
        LookupKey,
        [&ValueCB](const LookupKeyT &LookupKey, void *KeyStorage,
                   void *ValueStorage) -> std::pair<KeyT *, ValueT *> {
          KeyT *K = new (KeyStorage) KeyT(LookupKey);
          ValueT *V = new (ValueStorage) ValueT(ValueCB());
          return {K, V};
        });
  }

  template <typename LookupKeyT>
  InsertKVResultT
  update(const LookupKeyT &LookupKey,
         typename std::__type_identity<llvm::function_ref<std::pair<KeyT *, ValueT *>(
             const LookupKeyT &LookupKey, void *KeyStorage,
             void *ValueStorage)>>::type InsertCB,
         llvm::function_ref<ValueT &(KeyT &Key, ValueT &Value)> UpdateCB) {
    return map_internal::update<KeyT, ValueT>(LookupKey, InsertCB, UpdateCB,
                                              Info, SmallSize, getRawData(),
                                              getGroupsPtr());
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  typename std::enable_if<
      !std::is_same<ValueT, ValueCallbackT>::value &&
          std::is_same<ValueT,
                       decltype(std::declval<ValueCallbackT>()())>::value,

      InsertKVResultT>::type
  update(const LookupKeyT &LookupKey, ValueCallbackT ValueCB) {
    return update(
        LookupKey,
        [&ValueCB](const LookupKeyT &LookupKey, void *KeyStorage,
                   void *ValueStorage) -> std::pair<KeyT *, ValueT *> {
          KeyT *K = new (KeyStorage) KeyT(LookupKey);
          ValueT *V = new (ValueStorage) ValueT(ValueCB());
          return {K, V};
        },
        [&ValueCB](KeyT &Key, ValueT &Value) -> ValueT & {
          Value.~ValueT();
          return *new (&Value) ValueT(ValueCB());
        });
  }

  template <typename LookupKeyT>
  InsertKVResultT update(const LookupKeyT &LookupKey, ValueT NewV) {
    return update(
        LookupKey,
        [&NewV](const LookupKeyT &LookupKey, void *KeyStorage,
                void *ValueStorage) -> std::pair<KeyT *, ValueT *> {
          KeyT *K = new (KeyStorage) KeyT(LookupKey);
          ValueT *V = new (ValueStorage) ValueT(std::move(NewV));
          return {K, V};
        },
        [&NewV](KeyT &Key, ValueT &Value) -> ValueT & {
          Value.~ValueT();
          return *new (&Value) ValueT(std::move(NewV));
        });
  }

  template <typename LookupKeyT> bool erase(const LookupKeyT &LookupKey) {
    return map_internal::erase<KeyT, ValueT>(LookupKey, Info, SmallSize,
                                             getRawData());
  }

  template <typename CallbackT>
  void forEach(CallbackT Callback) {
    map_internal::forEach<KeyT, ValueT>(getSize(), getEntropy(), SmallSize,
                                        getRawData(), Callback);
  }

  void clear() {
    map_internal::clear<KeyT, ValueT>(Info, SmallSize, getRawData());
  }

  void reset() {
    map_internal::reset<KeyT, ValueT>(Info, SmallSize, &SmallBuffer, &Info,
                                      AllocatedGroups);
  }

  operator RefT() { return {Info, SmallSize, AllocatedGroups}; }

private:
  static constexpr int LinearSizeInPointer = sizeof(void *) / (sizeof(KeyT) + sizeof(ValueT));
  static constexpr int SmallSize =
      MinSmallSize < LinearSizeInPointer ? LinearSizeInPointer : MinSmallSize;
  static constexpr bool UseLinearLookup =
      map_internal::shouldUseLinearLookup<KeyT>(SmallSize);

  static_assert(SmallSize >= 0, "Cannot have a negative small size!");

  using GroupT = map_internal::Group<KeyT, ValueT>;

  map_internal::MapInfo Info;

  union {
    GroupT *AllocatedGroups;
    mutable map_internal::SmallSizeBuffer<KeyT, ValueT, UseLinearLookup, SmallSize> SmallBuffer;
  };

  int getSize() const { return Info.Size; }
  int getGrowthBudget() const { return Info.GrowthBudget; }
  int getEntropy() const { return Info.Entropy; }

  bool isSmall() const { return getEntropy() < 0; }

  void *getRawData() const {
    return !isSmall() ? (void *)AllocatedGroups : (void *)&SmallBuffer;
  }

  GroupT *&getGroupsPtr() {
    return AllocatedGroups;
  }
};

// Implementation of the routines in `map_internal` that are used above.
namespace map_internal {

template <typename KeyT> KeyT *getLinearKeys(void *Data) {
  return reinterpret_cast<KeyT *>(Data);
}

template <typename KeyT, typename LookupKeyT>
bool containsSmallLinear(const LookupKeyT &LookupKey, int Size, void *Data) {
  KeyT *Keys = getLinearKeys<KeyT>(Data);
  for (ssize_t i : llvm::seq<ssize_t>(0, static_cast<unsigned>(Size)))
    if (Keys[i] == LookupKey)
      return true;

  return false;
}

template <typename KeyT, typename ValueT>
ValueT *getLinearValues(void *Data, int SmallSize) {
  void *Values = getLinearKeys<KeyT>(Data) + static_cast<unsigned>(SmallSize);
  if (alignof(ValueT) > alignof(KeyT))
    Values = reinterpret_cast<void *>(llvm::alignAddr(Values, llvm::Align::Of<ValueT>()));
  return reinterpret_cast<ValueT *>(Values);
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
LookupKVResult<KeyT, ValueT> lookupSmallLinear(const LookupKeyT &LookupKey,
                                               int Size, int SmallSize,
                                               void *Data) {
  KeyT *Keys = getLinearKeys<KeyT>(Data);
  ValueT *Values = getLinearValues<KeyT, ValueT>(Data, SmallSize);
  for (ssize_t i : llvm::seq<ssize_t>(0, static_cast<unsigned>(Size)))
    if (Keys[i] == LookupKey)
      return {&Keys[i], &Values[i]};

  return {nullptr, nullptr};
}

inline size_t computeProbeMaskFromSize(ssize_t Size) {
  assert(llvm::isPowerOf2_64(Size) &&
         "Size must be a power of two for a hashed buffer!");
  // The probe mask needs to do too things: mask down to keep the index within
  // `Size`, and mask off the low bits to cause the index to start at the
  // beginning of a group. Since size is a power of two, this is equivalent to
  // `Size - 1` to get the Size-based mask, and masking it with
  // `~(GroupSize - 1)` to mask off the low bits. But we can fold these
  // two together as we know that `GroupSize` is also a power of two,
  // and thus this is equivalent to `(Size - 1) - (GroupSize - 1)`
  // which in turn is equivalent to the below value.
  return Size - 1;
}

/// This class handles building a sequence of probe indices from a given
/// starting point, including both the quadratic growth and masking the index
/// to stay within the bucket array size. The starting point doesn't need to be
/// clamped to the size ahead of time (or even by positive), we will do it
/// internally.
///
/// We compute the quadratic probe index incrementally, but we can also compute
/// it mathematically and will check that the incremental result matches our
/// mathematical expectation. We use the quadratic probing formula of:
///
///   p(x,s) = (x + (s + s^2)/2) mod Size
///
/// This particular quadratic sequence will visit every value modulo the
/// provided size.
class ProbeSequence {
  ssize_t Step = 0;
  size_t Mask;
  ssize_t i;
#ifndef NDEBUG
  ssize_t Start;
  ssize_t Size;
#endif

public:
  ProbeSequence(ssize_t Start, ssize_t Size) {
    Mask = computeProbeMaskFromSize(Size);
    i = Start & Mask;
#ifndef NDEBUG
    this->Start = Start;
    this->Size = Size;
#endif
  }

  void step() {
    ++Step;
    i = (i + Step) & Mask;
    assert(i == (((ssize_t)(Start & Mask) + (Step + Step * Step) / 2) % Size) &&
           "Index in probe sequence does not match the expected formula.");
    assert(Step < Size && "We necessarily visit all groups, so we can't have "
                          "more probe steps than groups.");
  }

  ssize_t getIndex() const { return i; }
};

inline uint8_t computeControlByte(size_t Hash) {
  // Mask one over the high bit so that engaged control bytes are easily
  // identified.
  return (Hash >> (sizeof(Hash) * 8 - 7)) | 0b10000000;
}

inline ssize_t computeHashIndex(size_t Hash, int Entropy) {
  return Hash ^ static_cast<unsigned>(Entropy);
}

template <typename LookupKeyT, typename GroupT>
std::tuple<GroupT *, ssize_t>
lookupIndexHashed(const LookupKeyT &LookupKey, int Entropy,
                  llvm::MutableArrayRef<GroupT> Groups) {
  //__asm volatile ("# LLVM-MCA-BEGIN hit");
  //__asm volatile ("# LLVM-MCA-BEGIN miss");
  size_t Hash = llvm::hash_value(LookupKey);
  uint8_t ControlByte = computeControlByte(Hash);
  ssize_t HashIndex = computeHashIndex(Hash, Entropy);

  ProbeSequence S(HashIndex, Groups.size());
  do {
    GroupT &G = Groups[S.getIndex()];
    auto ControlByteMatchedRange = G.match(ControlByte);
    if (LLVM_LIKELY(ControlByteMatchedRange)) {
      auto ByteIt = ControlByteMatchedRange.begin();
      auto ByteEnd = ControlByteMatchedRange.end();
      do {
        ssize_t ByteIndex = *ByteIt;
        if (LLVM_LIKELY(G.Keys[ByteIndex] == LookupKey))
          return {&G, ByteIndex};
        ++ByteIt;
      } while (LLVM_UNLIKELY(ByteIt != ByteEnd));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots and we're done probing.
    auto EmptyByteMatchedRange = G.matchEmpty();
    if (LLVM_LIKELY(EmptyByteMatchedRange)) {
      //__asm volatile("# LLVM-MCA-END miss");
      return {nullptr, 0};
    }

    S.step();
  } while (LLVM_UNLIKELY(true));
}

template <typename KeyT, typename ValueT, typename GroupT = Group<KeyT, ValueT>>
inline llvm::MutableArrayRef<GroupT> getGroups(void *Data, int Size) {
  return llvm::makeMutableArrayRef(reinterpret_cast<GroupT *>(Data), static_cast<unsigned>(Size));
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
bool containsHashed(const LookupKeyT &LookupKey, int Size, int Entropy,
                    void *Data) {
  auto Groups = getGroups<KeyT, ValueT>(Data, Size);
  return std::get<0>(lookupIndexHashed(LookupKey, Entropy, Groups)) != nullptr;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
LookupKVResult<KeyT, ValueT> lookupHashed(const LookupKeyT &LookupKey, int Size,
                                          int Entropy, void *Data) {
  using GroupT = Group<KeyT, ValueT>;
  llvm::MutableArrayRef<GroupT> Groups = getGroups<KeyT, ValueT>(Data, Size);

  GroupT *G;
  ssize_t ByteIndex;
  std::tie(G, ByteIndex) = lookupIndexHashed(LookupKey, Entropy, Groups);
  if (!G)
    return {nullptr, nullptr};

  return {&G->Keys[ByteIndex], &G->Values[ByteIndex]};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
bool contains(const LookupKeyT &LookupKey, int Size, int Entropy, int SmallSize,
              void *Data) {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(SmallSize) && Entropy < 0)
    return containsSmallLinear<KeyT>(LookupKey, Size, Data);

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return containsHashed<KeyT, ValueT>(LookupKey, Size, Entropy, Data);
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
LookupKVResult<KeyT, ValueT> lookup(const LookupKeyT &LookupKey, int Size,
                                    int Entropy, int SmallSize, void *Data) {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(SmallSize) && Entropy < 0)
    return lookupSmallLinear<KeyT, ValueT>(LookupKey, Size, SmallSize, Data);

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return lookupHashed<KeyT, ValueT>(LookupKey, Size, Entropy, Data);
}

template <typename KeyT, typename ValueT, typename CallbackT>
void forEachLinear(ssize_t Size, int SmallSize, void *Data, CallbackT Callback) {
    KeyT *Keys = getLinearKeys<KeyT>(Data);
    ValueT *Values = getLinearValues<KeyT, ValueT>(Data, SmallSize);
    for (ssize_t i : llvm::seq<ssize_t>(0, Size))
      Callback(Keys[i], Values[i]);
}

template <typename KeyT, typename ValueT, typename KVCallbackT,
          typename GroupCallbackT>
void forEachHashed(ssize_t Size, void *Data, KVCallbackT KVCallback,
                   GroupCallbackT GroupCallback) {
  using GroupT = Group<KeyT, ValueT>;
  llvm::MutableArrayRef<GroupT> Groups = getGroups<KeyT, ValueT>(Data, Size);
  for (GroupT &G : Groups) {
    auto PresentMatchedRange = G.matchPresent();
    if (!PresentMatchedRange)
      continue;
    for (ssize_t ByteIndex : PresentMatchedRange)
      KVCallback(G.Keys[ByteIndex], G.Values[ByteIndex]);

    GroupCallback(G);
  }
}

template <typename KeyT, typename ValueT, typename CallbackT>
void forEach(int InputSize, int Entropy, int SmallSize, void *Data,
             CallbackT Callback) {
  // We manually zero-extend size so that we can use simpler signed arithmetic
  // but not pay for a sign extension when doing pointer arithmetic.
  ssize_t Size = static_cast<unsigned>(InputSize);

  if (shouldUseLinearLookup<KeyT>(SmallSize) && Entropy < 0) {
    forEachLinear<KeyT, ValueT>(Size, SmallSize, Data, Callback);
    return;
  }

  // Otherwise walk the non-empty slots in each control group.
  forEachHashed<KeyT, ValueT>(Size, Data, Callback,
                              [](Group<KeyT, ValueT> &G) {});
}

template <typename GroupT>
class InsertIndexHashedResult {
public:
  InsertIndexHashedResult(GroupT *G, bool NeedsInsertion, ssize_t ByteIndex)
      : GroupAndNeedsInsertion(G, NeedsInsertion), ByteIndex(ByteIndex) {}

  GroupT *getGroup() { return GroupAndNeedsInsertion.getPointer(); }
  bool needsInsertion() { return GroupAndNeedsInsertion.getInt(); }
  ssize_t getByteIndex() { return ByteIndex; }

private:
  llvm::PointerIntPair<GroupT *, 1, bool> GroupAndNeedsInsertion;
  ssize_t ByteIndex;
};

// Tries to insert the given lookup key into the map. Returns three pieces of
// data compressed into two registers (in order to avoid an in-memory return).
// These are the group pointer, a bool representing whether insertion is in fact
// required, and the byte index of either the found entry in the group or the
// slot of the group to insert into. The group pointer will be null if insertion
// isn't possible without growing.
template <typename LookupKeyT, typename GroupT>
InsertIndexHashedResult<GroupT>
insertIndexHashed(const LookupKeyT &LookupKey, int GrowthBudget,
                  int Entropy, llvm::MutableArrayRef<GroupT> Groups) {
  size_t Hash = llvm::hash_value(LookupKey);
  uint8_t ControlByte = computeControlByte(Hash);
  ssize_t HashIndex = computeHashIndex(Hash, Entropy);

  GroupT *GroupWithDeleted = nullptr;
  typename GroupT::template MatchedByteRange<> DeletedMatchedRange;

  auto ReturnInsertAtIndex =
      [&](GroupT &G, ssize_t ByteIndex) -> InsertIndexHashedResult<GroupT> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    G.setByte(ByteIndex, ControlByte);
    return {&G, /*NeedsInsertion*/true, ByteIndex};
  };

  for (ProbeSequence S(HashIndex, Groups.size());; S.step()) {
    auto &G = Groups[S.getIndex()];

    auto ControlByteMatchedRange = G.match(ControlByte);
    if (LLVM_LIKELY(ControlByteMatchedRange)) {
      auto ByteIt = ControlByteMatchedRange.begin();
      auto ByteEnd = ControlByteMatchedRange.end();
      do {
        ssize_t ByteIndex = *ByteIt;
        if (LLVM_LIKELY(G.Keys[ByteIndex] == LookupKey))
          return {&G, /*NeedsInsertion=*/false, ByteIndex};
        ++ByteIt;
      } while (LLVM_UNLIKELY(ByteIt != ByteEnd));
    }

    // Track the first group with a deleted entry that we could insert over.
    if (!GroupWithDeleted) {
      DeletedMatchedRange = G.matchDeleted();
      if (DeletedMatchedRange)
        GroupWithDeleted = &G;
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // no empty slots. In that case, we'll continue probing.
    auto EmptyMatchedRange = G.matchEmpty();
    if (!EmptyMatchedRange)
      continue;

    // Ok, we've finished probing without finding anything and need to insert
    // instead.
    if (GroupWithDeleted)
      // If we found a deleted slot, we don't need the probe sequence to insert so just bail.
      break;

    // Otherwise, we're going to need to grow by inserting over one of these
    // empty slots. Check that we have the budget for that before we compute the
    // exact index of the empty slot. Without the growth budget we'll have to
    // completely rehash and so we can just bail here.
    if (GrowthBudget == 0)
      // Without room to grow, return that no group is viable but also set the
      // index to be negative. This ensures that a positive index is always
      // sufficient to determine that an existing was found.
      return {nullptr, /*NeedsInsertion*/true, 0};

    return ReturnInsertAtIndex(G, *EmptyMatchedRange.begin());
  }

  return ReturnInsertAtIndex(*GroupWithDeleted, *DeletedMatchedRange.begin());
}

template <typename LookupKeyT, typename GroupT>
std::pair<GroupT*, ssize_t> insertIntoEmptyIndex(
    const LookupKeyT& LookupKey, int Entropy, llvm::MutableArrayRef<GroupT> Groups) {
  size_t Hash = llvm::hash_value(LookupKey);
  uint8_t ControlByte = computeControlByte(Hash);
  ssize_t HashIndex = computeHashIndex(Hash, Entropy);

  for (ProbeSequence S(HashIndex, Groups.size());; S.step()) {
    auto &G = Groups[S.getIndex()];

    if (auto EmptyMatchedRange = G.matchEmpty()) {
      ssize_t FirstByteIndex = *EmptyMatchedRange.begin();
      G.setByte(FirstByteIndex, ControlByte);
      return {&G, FirstByteIndex};
    }

    // Otherwise we continue probing.
  }
}

inline int computeNewSize(int OldSize, bool IsOldLinear) {
  // If the old size is a linear size, scale it by the group size.
  if (IsOldLinear)
    OldSize /= GroupSize;

  if (OldSize < 4)
    // If we're going to heap allocate, get at least four groups.
    return 4;

  // Otherwise, we want the next power of two. This should always be a power of
  // two coming in, and so we just verify that. Also verify that this doesn't
  // overflow.
  assert(OldSize == (int)llvm::PowerOf2Ceil(OldSize) && "Expected a power of two!");
  return OldSize * 2;
}

inline int growthThresholdForSize(int Size) {
  // We use a 7/8ths load factor to trigger growth.
  Size *= GroupSize;
  return Size - Size / 8;
}

template <typename KeyT, typename ValueT, typename GroupT = Group<KeyT, ValueT>>
llvm::MutableArrayRef<GroupT> growAndRehash(MapInfo &Info,
                                      int SmallSize, void *Data,
                                      GroupT *&GroupsPtr) {
  // Capture the old values we will use early to make in unambiguous that they
  // aren't being updated.
  int OldSize = Info.Size;
  int OldEntropy = Info.Entropy;
  void *OldData = Data;

  bool IsOldLinear = shouldUseLinearLookup<KeyT>(SmallSize) && OldEntropy < 0;

  // Build up the new structure after growth without modifying the underlying
  // structures. This is important as some of those may be aliased by a small
  // buffer. We don't want to change the underlying state until we copy
  // everything out.
  int NewSize = computeNewSize(OldSize, IsOldLinear);
  GroupT *NewGroupsPtr = new GroupT[NewSize]();
  llvm::MutableArrayRef<GroupT> NewGroups(NewGroupsPtr, NewSize);
  int NewGrowthBudget = growthThresholdForSize(NewSize);
  int NewEntropy = llvm::hash_combine(OldEntropy, NewGroupsPtr) & EntropyMask;

  forEach<KeyT, ValueT>(
      OldSize, OldEntropy, SmallSize, OldData,
      [&](KeyT &OldKey, ValueT &OldValue) {
        GroupT *G;
        ssize_t ByteIndex;
        std::tie(G, ByteIndex) =
            insertIntoEmptyIndex(OldKey, NewEntropy, NewGroups);

        --NewGrowthBudget;
        new (&G->Keys[ByteIndex]) KeyT(std::move(OldKey));
        OldKey.~KeyT();
        new (&G->Values[ByteIndex]) ValueT(std::move(OldValue));
        OldValue.~ValueT();
      });
  assert(NewGrowthBudget >= 0 && "Must still have a growth budget after rehash!");

  if (OldEntropy >= 0) {
    // Old groups isn't a small buffer, so we need to deallocate it.
    delete[] getGroups<KeyT, ValueT>(OldData, OldSize).data();
  }

  // Now that we've fully built the new, grown structures, replace the entries
  // in the data structure. At this point we can be certain to not clobber
  // anything aliasing a small buffer.
  Info.Size = NewSize;
  Info.GrowthBudget = NewGrowthBudget;
  Info.Entropy = NewEntropy;
  GroupsPtr = NewGroupsPtr;

  // We return the newly allocated groups for immediate use by the caller.
  return NewGroups;
}

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT = Group<KeyT, ValueT>>
InsertKVResult<KeyT, ValueT> insertHashed(
    const LookupKeyT &LookupKey,
    llvm::function_ref<std::pair<KeyT *, ValueT *>(
        const LookupKeyT &LookupKey, void *KeyStorage, void *ValueStorage)>
        InsertCB,
    MapInfo &Info, int SmallSize, void *Data, GroupT *&GroupsPtr) {
  int &Size = Info.Size;
  int &GrowthBudget = Info.GrowthBudget;
  int &Entropy = Info.Entropy;
  llvm::MutableArrayRef<GroupT> Groups = getGroups<KeyT, ValueT>(Data, Size);

  auto Result = insertIndexHashed(LookupKey, GrowthBudget, Entropy, Groups);
  GroupT *G = Result.getGroup();
  ssize_t ByteIndex = Result.getByteIndex();
  if (!Result.needsInsertion()) {
    assert(G && "Must have a valid group when we find an existing entry.");
    return {false, G->Keys[ByteIndex], G->Values[ByteIndex]};
  }

  if (!G) {
    assert(
        GrowthBudget == 0 &&
        "Shouldn't need to grow the table until we exhaust our growth budget!");

    Groups =
        growAndRehash<KeyT, ValueT>(Info, SmallSize, Data, GroupsPtr);
    assert(GrowthBudget > 0 && "Must create growth budget after growing1");
    // Directly search for an empty index as we know we should have one.
    std::tie(G, ByteIndex) = insertIntoEmptyIndex(LookupKey, Entropy, Groups);
  }

  assert(G && "Should have a group to insert into now.");
  assert(ByteIndex >= 0 && "Should have a positive byte index now.");
  assert(GrowthBudget >= 0 && "Cannot insert with zero budget!");
  --GrowthBudget;

  KeyT *K;
  ValueT *V;
  std::tie(K, V) = InsertCB(LookupKey, &G->Keys[ByteIndex], &G->Values[ByteIndex]);
  return {true, *K, *V};
}

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT = Group<KeyT, ValueT>>
InsertKVResult<KeyT, ValueT> insertSmallLinear(
    const LookupKeyT &LookupKey,
    llvm::function_ref<std::pair<KeyT *, ValueT *>(
        const LookupKeyT &LookupKey, void *KeyStorage, void *ValueStorage)>
        InsertCB,
    MapInfo &Info, int SmallSize, void *Data, GroupT *&GroupsPtr) {
  KeyT *Keys = getLinearKeys<KeyT>(Data);
  ValueT *Values = getLinearValues<KeyT, ValueT>(Data, SmallSize);
  for (int i : llvm::seq(0, Info.Size))
    if (Keys[i] == LookupKey)
      return {false, Keys[i], Values[i]};

  KeyT *K;
  ValueT *V;

  // We need to insert. First see if we have space.
  if (Info.Size < SmallSize) {
    // We can do the easy linear insert.
    K = &Keys[Info.Size];
    V = &Values[Info.Size];
    ++Info.Size;
  } else {
    // No space for a linear insert so grow into a hash table and then do
    // a hashed insert.
    llvm::MutableArrayRef<GroupT> Groups =
        growAndRehash<KeyT, ValueT>(Info, SmallSize, Data, GroupsPtr);
    assert(Info.GrowthBudget > 0 && "Must create growth budget after growing1");
    GroupT *G;
    int ByteIndex;
    std::tie(G, ByteIndex) =
        insertIntoEmptyIndex(LookupKey, Info.Entropy, Groups);
    --Info.GrowthBudget;
    K = &G->Keys[ByteIndex];
    V = &G->Values[ByteIndex];
  }
  std::tie(K, V) = InsertCB(LookupKey, K, V);
  return {true, *K, *V};
}

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT>
InsertKVResult<KeyT, ValueT>
insert(const LookupKeyT &LookupKey,
       llvm::function_ref<std::pair<KeyT *, ValueT *>(
           const LookupKeyT &LookupKey, void *KeyStorage, void *ValueStorage)>
           InsertCB,
       MapInfo &Info, int SmallSize, void *Data, GroupT *&GroupsPtr) {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(SmallSize) && Info.Entropy < 0)
    return insertSmallLinear<KeyT, ValueT>(LookupKey, InsertCB, Info, SmallSize,
                                           Data, GroupsPtr);

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return insertHashed<KeyT, ValueT>(LookupKey, InsertCB, Info, SmallSize, Data,
                                    GroupsPtr);
}

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT>
InsertKVResult<KeyT, ValueT>
update(const LookupKeyT &LookupKey,
       llvm::function_ref<std::pair<KeyT *, ValueT *>(
           const LookupKeyT &LookupKey, void *KeyStorage, void *ValueStorage)>
           InsertCB,
       llvm::function_ref<ValueT &(KeyT &Key, ValueT &Value)> UpdateCB, MapInfo &Info,
       int SmallSize, void *Data, GroupT *&GroupsPtr) {
  auto IResult = insert(LookupKey, InsertCB, Info, SmallSize, Data, GroupsPtr);

  if (IResult.isInserted())
    return IResult;

  ValueT &V = UpdateCB(IResult.getKey(), IResult.getValue());
  return {false, IResult.getKey(), V};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
bool eraseSmallLinear(const LookupKeyT &LookupKey, int &Size, int SmallSize,
                      void *Data) {
  KeyT *Keys = getLinearKeys<KeyT>(Data);
  ValueT *Values = getLinearValues<KeyT, ValueT>(Data, SmallSize);
  for (int i : llvm::seq(0, Size))
    if (Keys[i] == LookupKey) {
      // Found the key, so clobber this entry with the last one and decrease
      // the size by one.
      --Size;
      Keys[i] = std::move(Keys[Size]);
      Keys[Size].~KeyT();
      Values[i] = std::move(Values[Size]);
      Values[Size].~ValueT();
      return true;
    }

  return false;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
bool eraseHashed(const LookupKeyT &LookupKey, MapInfo &Info, void *Data) {
  using GroupT = Group<KeyT, ValueT>;
  llvm::MutableArrayRef<GroupT> Groups = getGroups<KeyT, ValueT>(Data, Info.Size);

  GroupT *G;
  ssize_t ByteIndex;
  std::tie(G, ByteIndex) = lookupIndexHashed(LookupKey, Info.Entropy, Groups);
  if (!G)
    return false;

  G->Keys[ByteIndex].~KeyT();
  G->Values[ByteIndex].~ValueT();

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  auto EmptyMatchedRange = G->matchEmpty();
  if (EmptyMatchedRange) {
    G->setByte(ByteIndex, GroupT::Empty);
    ++Info.GrowthBudget;
  } else {
    G->setByte(ByteIndex, GroupT::Deleted);
  }
  return true;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
bool erase(const LookupKeyT &LookupKey, MapInfo &Info, int SmallSize,
           void *Data) {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear erases
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(SmallSize) && Info.Entropy < 0)
    return eraseSmallLinear<KeyT, ValueT>(LookupKey, Info.Size, SmallSize,
                                          Data);

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return eraseHashed<KeyT, ValueT>(LookupKey, Info, Data);
}

template <typename KeyT, typename ValueT>
void clear(MapInfo &Info, int SmallSize, void *Data) {
  // We manually zero-extend size so that we can use simpler signed arithmetic
  // but not pay for a sign extension when doing pointer arithmetic.
  ssize_t Size = static_cast<unsigned>(Info.Size);

  auto DestroyCB = [](KeyT &K, ValueT &V) {
    // Destroy this key and value.
    K.~KeyT();
    V.~ValueT();
  };

  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(SmallSize) && Info.Entropy < 0) {
    // Destroy all the keys and values.
    forEachLinear<KeyT, ValueT>(Size, SmallSize, Data, DestroyCB);

    // Now reset the size to zero and we'll start again inserting into the
    // beginning of the small linear buffer.
    Info.Size = 0;
    return;
  }

  // Otherwise walk the non-empty slots in the control group destroying each
  // one and clearing out the group.
  forEachHashed<KeyT, ValueT>(Size, Data, DestroyCB,
                              [](Group<KeyT, ValueT> &G) {
                                // Clear the group.
                                G.clear();
                              });

  // And reset the growth budget.
  Info.GrowthBudget = growthThresholdForSize(Info.Size);
}

template <typename KeyT, typename ValueT>
void init(MapInfo &Info, int SmallSize, void *SmallBufferAddr, void *ThisAddr) {
  Info.Entropy = llvm::hash_value(ThisAddr) & EntropyMask;
  // Mark that we don't have external storage allocated.
  Info.Entropy = -Info.Entropy;

  if (map_internal::shouldUseLinearLookup<KeyT>(SmallSize)) {
    // We use size to mean empty when doing linear lookups.
    Info.Size = 0;
    // Growth budget isn't relevant as long as we're doing linear lookups.
    Info.GrowthBudget = 0;
    return;
  }

  // We're not using linear lookups in the small size, so initialize it as
  // an initial hash table.
  Info.Size = SmallSize / GroupSize;
  Info.GrowthBudget = map_internal::growthThresholdForSize(Info.Size);
  auto Groups = getGroups<KeyT, ValueT>(SmallBufferAddr, Info.Size);
  for (auto &G : Groups)
    G.clear();
}

template <typename KeyT, typename ValueT, typename GroupT>
void reset(MapInfo &Info, int SmallSize, void *SmallBufferAddr, void *ThisAddr,
           GroupT *GroupsPtr) {
  // If we have a small size and are still using it, this is just clearing.
  if (Info.Entropy < 0) {
    clear<KeyT, ValueT>(Info, SmallSize, SmallBufferAddr);
    return;
  }

  // Otherwise do the first part of the clear to destroy all the elements.
  forEachHashed<KeyT, ValueT>(
      static_cast<unsigned>(Info.Size), GroupsPtr,
      [](KeyT &K, ValueT &V) {
        K.~KeyT();
        V.~ValueT();
      },
      [](Group<KeyT, ValueT> &G) {});

  // Deallocate the buffer.
  delete[] GroupsPtr;

  // Re-initialize the whole thing.
  init<KeyT, ValueT>(Info, SmallSize, SmallBufferAddr, ThisAddr);
}

template <typename KeyT, typename ValueT>
void copyInit(MapInfo &Info, int SmallSize, void *SmallBufferAddr, void *ThisAddr,
              const MapInfo &ArgInfo,  void *ArgSmallBufferAddr, void *ArgAddr) {
}

} // namespace map_internal
}  // namespace Carbon

#endif
