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

template <typename KeyT, typename ValueT>
class MapView;
template <typename KeyT, typename ValueT>
class MapRef;
template <typename KeyT, typename ValueT, int MinSmallSize>
class Map;

namespace MapInternal {

// Detect whether we can use SIMD accelerated implementations of the control
// groups.
#if 1 || defined(__SSSE3__)
#define CARBON_USE_SSE_CONTROL_GROUP 1
#if 1 || defined(__SSE4_1__)
#define CARBON_OPTIMIZE_SSE4_1 1
#endif
#endif

struct MapInfo {
  int Size;
  int GrowthBudget;
  int Entropy;
};

template <typename KeyT, typename ValueT>
class LookupKVResult {
 public:
  LookupKVResult() = default;
  LookupKVResult(KeyT* key, ValueT* value) : Key(key), Value(value) {}

  explicit operator bool() const { return Key != nullptr; }

  auto getKey() const -> KeyT& { return *Key; }
  auto getValue() const -> ValueT& { return *Value; }

 private:
  KeyT* Key = nullptr;
  ValueT* Value = nullptr;
};

template <typename KeyT, typename ValueT>
class InsertKVResult {
 public:
  InsertKVResult() = default;
  InsertKVResult(bool inserted, KeyT& key, ValueT& value)
      : KeyAndInserted(&key, inserted), Value(&value) {}

  auto isInserted() const -> bool { return KeyAndInserted.getInt(); }

  auto getKey() const -> KeyT& { return *KeyAndInserted.getPointer(); }
  auto getValue() const -> ValueT& { return *Value; }

 private:
  llvm::PointerIntPair<KeyT*, 1, bool> KeyAndInserted;
  ValueT* Value = nullptr;
};

// We organize the hashtable into 16-slot groups so that we can use 16 control
// bytes to understand the contents efficiently.
constexpr int GroupSize = 16;

template <typename KeyT, typename ValueT>
struct Group;
template <typename KeyT, typename ValueT, bool IsCmpVec = true>
class GroupMatchedByteRange;

template <typename KeyT, typename ValueT>
class GroupMatchedByteIterator
    : public llvm::iterator_facade_base<GroupMatchedByteIterator<KeyT, ValueT>,
                                        std::forward_iterator_tag, int, int> {
  friend struct Group<KeyT, ValueT>;
  friend class GroupMatchedByteRange<KeyT, ValueT, /*IsCmpVec=*/true>;
  friend class GroupMatchedByteRange<KeyT, ValueT, /*IsCmpVec=*/false>;

  ssize_t byte_index;

  unsigned mask = 0;

  explicit GroupMatchedByteIterator(unsigned mask) : mask(mask) {}

 public:
  GroupMatchedByteIterator() = default;

  auto operator==(const GroupMatchedByteIterator& rhs) const -> bool {
    return mask == rhs.mask;
  }

  auto operator*() -> ssize_t& {
    assert(mask != 0 && "Cannot get an index from a zero mask!");
    byte_index = llvm::countTrailingZeros(mask, llvm::ZB_Undefined);
    return byte_index;
  }

  auto operator++() -> GroupMatchedByteIterator& {
    assert(mask != 0 && "Must not be called with a zero mask!");
    mask &= (mask - 1);
    return *this;
  }
};

template <typename KeyT, typename ValueT, bool IsCmpVec>
class GroupMatchedByteRange {
  friend struct Group<KeyT, ValueT>;
  using MatchedByteIterator = GroupMatchedByteIterator<KeyT, ValueT>;

#if CARBON_OPTIMIZE_SSE4_1
  __m128i mask_vec;

  explicit GroupMatchedByteRange(__m128i mask_vec) : mask_vec(mask_vec) {}
#else
  unsigned Mask;

  explicit GroupMatchedByteRange(unsigned mask) : Mask(mask) {}
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
  auto empty() const -> bool {
#if CARBON_OPTIMIZE_SSE4_1
    return _mm_test_all_zeros(
        mask_vec,
        IsCmpVec ? mask_vec : _mm_set1_epi8(static_cast<char>(0b10000000U)));
#else
    return Mask == 0;
#endif
  }

  auto begin() const -> MatchedByteIterator {
#if CARBON_OPTIMIZE_SSE4_1
    unsigned mask = _mm_movemask_epi8(mask_vec);
    return MatchedByteIterator(mask);
#else
    return MatchedByteIterator(Mask);
#endif
  }

  auto end() const -> MatchedByteIterator { return MatchedByteIterator(); }
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
  __m128i byte_vec = {};
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

  auto match(uint8_t match_byte) const -> MatchedByteRange<> {
#if CARBON_USE_SSE_CONTROL_GROUP
    auto match_byte_vec = _mm_set1_epi8(match_byte);
    auto match_byte_cmp_vec = _mm_cmpeq_epi8(byte_vec, match_byte_vec);
#if CARBON_OPTIMIZE_SSE4_1
    return MatchedByteRange<>(match_byte_cmp_vec);
#else
    return MatchedByteRange<>((unsigned)_mm_movemask_epi8(MatchByteCmpVec));
#endif
#else
    unsigned mask = 0;
    for (int byte_index : llvm::seq(0, GroupSize)) {
      mask |= (Bytes[byte_index] == match_byte) << byte_index;
    }
    return MatchedByteRange<>(mask);
#endif
  }

  auto matchEmpty() const -> MatchedByteRange<> { return match(Empty); }

  auto matchDeleted() const -> MatchedByteRange<> { return match(Deleted); }

  auto matchPresent() const -> MatchedByteRange</*IsCmpVec=*/false> {
#if CARBON_USE_SSE_CONTROL_GROUP
#if CARBON_OPTIMIZE_SSE4_1
    return MatchedByteRange</*IsCmpVec=*/false>(byte_vec);
#else
    // We arrange the byte vector for present bytes so that we can directly
    // extract it as a mask.
    return MatchedByteRange</*IsCmpVec=*/false>((unsigned)_mm_movemask_epi8(ByteVec));
#endif
#else
    // Generic code to compute a bitmask.
    unsigned mask = 0;
    for (int byte_index : llvm::seq(0, GroupSize)) {
      mask |= static_cast<bool>(Bytes[byte_index] & 0b10000000U) << byte_index;
    }
    return MatchedByteRange</*IsCmpVec=*/false>(mask);
#endif
  }

  void setByte(int byte_index, uint8_t control_byte) {
#if CARBON_USE_SSE_CONTROL_GROUP
    // We directly manipulate the storage of the vector as there isn't a nice
    // intrinsic for this.
    reinterpret_cast<unsigned char*>(&byte_vec)[byte_index] = control_byte;
#else
    Bytes[byte_index] = control_byte;
#endif
  }

  void clear() {
#if CARBON_USE_SSE_CONTROL_GROUP
    byte_vec = _mm_set1_epi8(Empty);
#else
    for (int byte_index : llvm::seq(0, GroupSize)) {
      Bytes[byte_index] = Empty;
    }
#endif
  }
};

constexpr int EntropyMask = INT_MAX & ~(GroupSize - 1);

template <typename KeyT>
constexpr auto getKeysInCacheline() -> int {
  constexpr int CachelineSize = 64;

  return CachelineSize / sizeof(KeyT);
}

template <typename KeyT>
constexpr auto shouldUseLinearLookup(int small_size) -> bool {
  return small_size <= getKeysInCacheline<KeyT>();
}

template <typename KeyT, typename ValueT, bool UseLinearLookup, int SmallSize>
struct SmallSizeBuffer;

template <typename KeyT, typename ValueT>
struct SmallSizeBuffer<KeyT, ValueT, true, 0> {};

template <typename KeyT, typename ValueT, int SmallSize>
struct SmallSizeBuffer<KeyT, ValueT, true, SmallSize> {
  union {
    KeyT Keys[SmallSize];
  };
  union {
    ValueT Values[SmallSize];
  };
};

template <typename KeyT, typename ValueT, int SmallSize>
struct SmallSizeBuffer<KeyT, ValueT, false, SmallSize> {
  using GroupT = MapInternal::Group<KeyT, ValueT>;

  // FIXME: One interesting question is whether the small size should be a
  // minumum here or an exact figure.
  static_assert(llvm::isPowerOf2_32(SmallSize),
                "SmallSize must be a power of two for a hashed buffer!");
  static_assert(SmallSize >= MapInternal::GroupSize,
                "SmallSize must be at least the size of one group!");
  static_assert((SmallSize % MapInternal::GroupSize) == 0,
                "SmallSize must be a multiple of the group size!");
  static constexpr int SmallNumGroups = SmallSize / MapInternal::GroupSize;
  static_assert(llvm::isPowerOf2_32(SmallNumGroups),
                "The number of groups must be a power of two when hashing!");

  GroupT Groups[SmallNumGroups];
};

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto contains(const LookupKeyT& lookup_key, int size, int entropy,
              int small_size, void* data) -> bool;

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookup(const LookupKeyT& lookup_key, int size, int entropy, int small_size,
            void* data) -> LookupKVResult<KeyT, ValueT>;

template <typename KeyT, typename ValueT, typename CallbackT>
void forEach(int size, int entropy, int small_size, void* data,
             CallbackT callback);

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT = Group<KeyT, ValueT>>
auto insert(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    MapInfo& info, int small_size, void* data, GroupT*& groups_ptr)
    -> InsertKVResult<KeyT, ValueT>;

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT = Group<KeyT, ValueT>>
auto update(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb,
    MapInfo& info, int small_size, void* data, GroupT*& groups_ptr)
    -> InsertKVResult<KeyT, ValueT>;

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto erase(const LookupKeyT& lookup_key, MapInfo& info, int small_size,
           void* data) -> bool;

template <typename KeyT, typename ValueT>
void clear(MapInfo& info, int small_size, void* data);

template <typename KeyT, typename ValueT>
void init(MapInfo& info, int small_size, void* small_buffer_addr,
          void* this_addr);

template <typename KeyT, typename ValueT, typename GroupT = Group<KeyT, ValueT>>
void reset(MapInfo& info, int small_size, void* small_buffer_addr,
           void* this_addr, GroupT* groups_ptr);

template <typename KeyT, typename ValueT>
void copyInit(MapInfo& info, int small_size, void* small_buffer_addr,
              void* this_addr, const MapInfo& arg_info,
              void* arg_small_buffer_addr, void* arg_addr);

}  // namespace MapInternal

template <typename InputKeyT, typename InputValueT>
class MapView {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using LookupKVResultT = typename MapInternal::LookupKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  auto contains(const LookupKeyT& lookup_key) const -> bool {
    return MapInternal::contains<KeyT, ValueT>(lookup_key, Size, Entropy,
                                               SmallSize, Data);
  }

  template <typename LookupKeyT>
  auto lookup(const LookupKeyT& lookup_key) const -> LookupKVResultT {
    return MapInternal::lookup<KeyT, ValueT>(lookup_key, Size, Entropy,
                                             SmallSize, Data);
  }

  template <typename LookupKeyT>
  auto operator[](const LookupKeyT& lookup_key) const -> ValueT* {
    auto result = lookup(lookup_key);
    return result ? &result.getValue() : nullptr;
  }

  template <typename CallbackT>
  void forEach(CallbackT callback) {
    MapInternal::forEach<KeyT, ValueT>(Size, Entropy, SmallSize, Data,
                                       callback);
  }

 private:
  template <typename MapKeyT, typename MapValueT, int MinSmallSize>
  friend class Map;
  friend class MapRef<KeyT, ValueT>;

  MapView(int size, int entropy, int small_size, void* data)
      : Size(size), Entropy(entropy), SmallSize(small_size), Data(data) {}

  int Size;
  int Entropy;
  int SmallSize;
  void* Data;

  template <typename LookupKeyT>
  static auto contains(const LookupKeyT& lookup_key, int size, int entropy,
                       int small_size, void* data) -> bool;

  template <typename LookupKeyT>
  static auto lookup(const LookupKeyT& lookup_key, int size, int entropy,
                     int small_size, void* data) -> LookupKVResultT;
};

template <typename InputKeyT, typename InputValueT>
class MapRef {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using LookupKVResultT = MapInternal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = MapInternal::InsertKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  auto contains(const LookupKeyT& lookup_key) const -> bool {
    return MapInternal::contains<KeyT, ValueT>(
        lookup_key, getSize(), getEntropy(), SmallSize, getRawData());
  }

  template <typename LookupKeyT>
  auto lookup(const LookupKeyT& lookup_key) const -> LookupKVResultT {
    return MapInternal::lookup<KeyT, ValueT>(
        lookup_key, getSize(), getEntropy(), SmallSize, getRawData());
  }

  template <typename LookupKeyT>
  auto operator[](const LookupKeyT& lookup_key) const -> ValueT* {
    auto result = lookup(lookup_key);
    return result ? &result.getValue() : nullptr;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const {
    return {getSize(), getEntropy(), SmallSize, getRawData()};
  }

  template <typename LookupKeyT>
  auto insert(
      const LookupKeyT& lookup_key,
      typename std::__type_identity<llvm::function_ref<std::pair<
          KeyT*, ValueT*>(const LookupKeyT& lookup_key, void* key_storage,
                          void* value_storage)>>::type insert_cb)
      -> InsertKVResultT {
    return MapInternal::insert<KeyT, ValueT, LookupKeyT>(
        lookup_key, insert_cb, *Info, SmallSize, getRawData(), getGroupsPtr());
  }

  template <typename LookupKeyT>
  auto insert(const LookupKeyT& lookup_key, ValueT new_v) -> InsertKVResultT {
    return insert(lookup_key,
                  [&new_v](const LookupKeyT& lookup_key, void* key_storage,
                           void* value_storage) -> std::pair<KeyT*, ValueT*> {
                    KeyT* k = new (key_storage) KeyT(lookup_key);
                    auto* v = new (value_storage) ValueT(std::move(new_v));
                    return {k, v};
                  });
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  auto insert(const LookupKeyT& lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResultT>::type {
    return insert(
        lookup_key,
        [&value_cb](const LookupKeyT& lookup_key, void* key_storage,
                    void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(value_cb());
          return {k, v};
        });
  }

  template <typename LookupKeyT>
  auto update(
      const LookupKeyT& lookup_key,
      typename std::__type_identity<llvm::function_ref<std::pair<
          KeyT*, ValueT*>(const LookupKeyT& lookup_key, void* key_storage,
                          void* value_storage)>>::type insert_cb,
      llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
      -> InsertKVResultT {
    return MapInternal::update<KeyT, ValueT>(lookup_key, insert_cb, update_cb,
                                             *Info, SmallSize, getRawData(),
                                             getGroupsPtr());
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  auto update(const LookupKeyT& lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResultT>::type {
    return update(
        lookup_key,
        [&value_cb](const LookupKeyT& lookup_key, void* key_storage,
                    void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(value_cb());
          return {k, v};
        },
        [&value_cb](KeyT& /*Key*/, ValueT& value) -> ValueT& {
          value.~ValueT();
          return *new (&value) ValueT(value_cb());
        });
  }

  template <typename LookupKeyT>
  auto update(const LookupKeyT& lookup_key, ValueT new_v) -> InsertKVResultT {
    return update(
        lookup_key,
        [&new_v](const LookupKeyT& lookup_key, void* key_storage,
                 void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(std::move(new_v));
          return {k, v};
        },
        [&new_v](KeyT& /*Key*/, ValueT& value) -> ValueT& {
          value.~ValueT();
          return *new (&value) ValueT(std::move(new_v));
        });
  }

  template <typename LookupKeyT>
  auto erase(const LookupKeyT& lookup_key) -> bool {
    return MapInternal::erase<KeyT, ValueT>(lookup_key, *Info, SmallSize,
                                            getRawData());
  }

  template <typename CallbackT>
  void forEach(CallbackT callback) {
    MapInternal::forEach<KeyT, ValueT>(getSize(), getEntropy(), SmallSize,
                                       getRawData(), callback);
  }

  void clear() {
    MapInternal::clear<KeyT, ValueT>(*Info, SmallSize, getRawData());
  }

  void reset() {
    MapInternal::reset<KeyT, ValueT>(*Info, SmallSize, getSmallBufferAddr(),
                                     Info, getGroupsPtr());
  }

 private:
  template <typename MapKeyT, typename MapValueT, int MinSmallSize>
  friend class Map;

  using GroupT = MapInternal::Group<KeyT, ValueT>;

  MapRef(MapInternal::MapInfo& info, int small_size, GroupT*& groups_ptr)
      : Info(&info),
        SmallSize(small_size),
        GroupsPtrOrSmallBuffer(&groups_ptr) {}

  MapInternal::MapInfo* Info;
  int SmallSize;
  GroupT** GroupsPtrOrSmallBuffer;

  auto getSize() const -> int { return Info->Size; }
  auto getGrowthBudget() const -> int { return Info->GrowthBudget; }
  auto getEntropy() const -> int { return Info->Entropy; }

  auto getSmallBufferAddr() const -> void* { return GroupsPtrOrSmallBuffer; }
  auto getGroupsPtr() const -> GroupT*& {
    return *reinterpret_cast<GroupT**>(GroupsPtrOrSmallBuffer);
  }

  auto isSmall() const -> bool { return getEntropy() < 0; }

  auto getRawData() const -> void* {
    return isSmall() ? getSmallBufferAddr() : getGroupsPtr();
  }
};

template <typename InputKeyT, typename InputValueT, int MinSmallSize = 0>
class Map {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using RefT = MapRef<KeyT, ValueT>;
  using LookupKVResultT = MapInternal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = MapInternal::InsertKVResult<KeyT, ValueT>;

  Map() {
    MapInternal::init<KeyT, ValueT>(Info, SmallSize, &SmallBuffer, &Info);
  }
  ~Map() {
    if (!isSmall()) {
      // Not using a small buffer, so we need to deallocate it.
      delete[] AllocatedGroups;
    }
  }
  Map(const Map& arg) {
    MapInternal::copyInit<KeyT, ValueT>(Info, SmallSize, &SmallBuffer, &Info,
                                        arg.Info, &arg.SmallBuffer, &arg.Info);
  }
  Map(Map&& arg) = delete;

  template <typename LookupKeyT>
  auto contains(const LookupKeyT& lookup_key) const -> bool {
    return MapInternal::contains<KeyT, ValueT>(
        lookup_key, getSize(), getEntropy(), SmallSize, getRawData());
  }

  template <typename LookupKeyT>
  auto lookup(const LookupKeyT& lookup_key) const -> LookupKVResultT {
    return MapInternal::lookup<KeyT, ValueT>(
        lookup_key, getSize(), getEntropy(), SmallSize, getRawData());
  }

  template <typename LookupKeyT>
  auto operator[](const LookupKeyT& lookup_key) const -> ValueT* {
    auto result = lookup(lookup_key);
    return result ? &result.getValue() : nullptr;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const {
    return {getSize(), getEntropy(), SmallSize, getRawData()};
  }

  template <typename LookupKeyT>
  auto insert(
      const LookupKeyT& lookup_key,
      typename std::__type_identity<llvm::function_ref<std::pair<
          KeyT*, ValueT*>(const LookupKeyT& lookup_key, void* key_storage,
                          void* value_storage)>>::type insert_cb)
      -> InsertKVResultT {
    return MapInternal::insert<KeyT, ValueT, LookupKeyT>(
        lookup_key, insert_cb, Info, SmallSize, getRawData(), getGroupsPtr());
  }

  template <typename LookupKeyT>
  auto insert(const LookupKeyT& lookup_key, ValueT new_v) -> InsertKVResultT {
    return insert(lookup_key,
                  [&new_v](const LookupKeyT& lookup_key, void* key_storage,
                           void* value_storage) -> std::pair<KeyT*, ValueT*> {
                    KeyT* k = new (key_storage) KeyT(lookup_key);
                    auto* v = new (value_storage) ValueT(std::move(new_v));
                    return {k, v};
                  });
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  auto insert(const LookupKeyT& lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResultT>::type {
    return insert(
        lookup_key,
        [&value_cb](const LookupKeyT& lookup_key, void* key_storage,
                    void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(value_cb());
          return {k, v};
        });
  }

  template <typename LookupKeyT>
  auto update(
      const LookupKeyT& lookup_key,
      typename std::__type_identity<llvm::function_ref<std::pair<
          KeyT*, ValueT*>(const LookupKeyT& lookup_key, void* key_storage,
                          void* value_storage)>>::type insert_cb,
      llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
      -> InsertKVResultT {
    return MapInternal::update<KeyT, ValueT>(lookup_key, insert_cb, update_cb,
                                             Info, SmallSize, getRawData(),
                                             getGroupsPtr());
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  auto update(const LookupKeyT& lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResultT>::type {
    return update(
        lookup_key,
        [&value_cb](const LookupKeyT& lookup_key, void* key_storage,
                    void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(value_cb());
          return {k, v};
        },
        [&value_cb](KeyT& /*Key*/, ValueT& value) -> ValueT& {
          value.~ValueT();
          return *new (&value) ValueT(value_cb());
        });
  }

  template <typename LookupKeyT>
  auto update(const LookupKeyT& lookup_key, ValueT new_v) -> InsertKVResultT {
    return update(
        lookup_key,
        [&new_v](const LookupKeyT& lookup_key, void* key_storage,
                 void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(std::move(new_v));
          return {k, v};
        },
        [&new_v](KeyT& /*Key*/, ValueT& value) -> ValueT& {
          value.~ValueT();
          return *new (&value) ValueT(std::move(new_v));
        });
  }

  template <typename LookupKeyT>
  auto erase(const LookupKeyT& lookup_key) -> bool {
    return MapInternal::erase<KeyT, ValueT>(lookup_key, Info, SmallSize,
                                            getRawData());
  }

  template <typename CallbackT>
  void forEach(CallbackT callback) {
    MapInternal::forEach<KeyT, ValueT>(getSize(), getEntropy(), SmallSize,
                                       getRawData(), callback);
  }

  void clear() {
    MapInternal::clear<KeyT, ValueT>(Info, SmallSize, getRawData());
  }

  void reset() {
    MapInternal::reset<KeyT, ValueT>(Info, SmallSize, &SmallBuffer, &Info,
                                     AllocatedGroups);
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator RefT() { return {Info, SmallSize, AllocatedGroups}; }

 private:
  static constexpr int LinearSizeInPointer =
      sizeof(void*) / (sizeof(KeyT) + sizeof(ValueT));
  static constexpr int SmallSize =
      MinSmallSize < LinearSizeInPointer ? LinearSizeInPointer : MinSmallSize;
  static constexpr bool UseLinearLookup =
      MapInternal::shouldUseLinearLookup<KeyT>(SmallSize);

  static_assert(SmallSize >= 0, "Cannot have a negative small size!");

  using GroupT = MapInternal::Group<KeyT, ValueT>;

  MapInternal::MapInfo Info;

  union {
    GroupT* AllocatedGroups;
    mutable MapInternal::SmallSizeBuffer<KeyT, ValueT, UseLinearLookup,
                                         SmallSize>
        SmallBuffer;
  };

  auto getSize() const -> int { return Info.Size; }
  auto getGrowthBudget() const -> int { return Info.GrowthBudget; }
  auto getEntropy() const -> int { return Info.Entropy; }

  auto isSmall() const -> bool { return getEntropy() < 0; }

  auto getRawData() const -> void* {
    return !isSmall() ? static_cast<void*>(AllocatedGroups)
                      : static_cast<void*>(&SmallBuffer);
  }

  auto getGroupsPtr() -> GroupT*& { return AllocatedGroups; }
};

// Implementation of the routines in `map_internal` that are used above.
namespace MapInternal {

template <typename KeyT>
auto getLinearKeys(void* data) -> KeyT* {
  return reinterpret_cast<KeyT*>(data);
}

template <typename KeyT, typename LookupKeyT>
auto containsSmallLinear(const LookupKeyT& lookup_key, int size, void* data)
    -> bool {
  KeyT* keys = getLinearKeys<KeyT>(data);
  for (ssize_t i : llvm::seq<ssize_t>(0, static_cast<unsigned>(size))) {
    if (keys[i] == lookup_key) {
      return true;
    }
  }

  return false;
}

template <typename KeyT, typename ValueT>
auto getLinearValues(void* data, int small_size) -> ValueT* {
  void* values = getLinearKeys<KeyT>(data) + static_cast<unsigned>(small_size);
  if (alignof(ValueT) > alignof(KeyT)) {
    values = reinterpret_cast<void*>(
        llvm::alignAddr(values, llvm::Align::Of<ValueT>()));
  }
  return reinterpret_cast<ValueT*>(values);
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookupSmallLinear(const LookupKeyT& lookup_key, int size, int small_size,
                       void* data) -> LookupKVResult<KeyT, ValueT> {
  KeyT* keys = getLinearKeys<KeyT>(data);
  ValueT* values = getLinearValues<KeyT, ValueT>(data, small_size);
  for (ssize_t i : llvm::seq<ssize_t>(0, static_cast<unsigned>(size))) {
    if (keys[i] == lookup_key) {
      return {&keys[i], &values[i]};
    }
  }

  return {nullptr, nullptr};
}

inline auto computeProbeMaskFromSize(ssize_t size) -> size_t {
  assert(llvm::isPowerOf2_64(size) &&
         "Size must be a power of two for a hashed buffer!");
  // The probe mask needs to do too things: mask down to keep the index within
  // `Size`, and mask off the low bits to cause the index to start at the
  // beginning of a group. Since size is a power of two, this is equivalent to
  // `Size - 1` to get the Size-based mask, and masking it with
  // `~(GroupSize - 1)` to mask off the low bits. But we can fold these
  // two together as we know that `GroupSize` is also a power of two,
  // and thus this is equivalent to `(Size - 1) - (GroupSize - 1)`
  // which in turn is equivalent to the below value.
  return size - 1;
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
  ProbeSequence(ssize_t start, ssize_t size) {
    Mask = computeProbeMaskFromSize(size);
    i = start & Mask;
#ifndef NDEBUG
    this->Start = start;
    this->Size = size;
#endif
  }

  void step() {
    ++Step;
    i = (i + Step) & Mask;
    assert(i == (((ssize_t)(Start & Mask) + (Step + Step * Step) / 2) % Size) &&
           "Index in probe sequence does not match the expected formula.");
    assert(Step < Size &&
           "We necessarily visit all groups, so we can't have "
           "more probe steps than groups.");
  }

  auto getIndex() const -> ssize_t { return i; }
};

inline auto computeControlByte(size_t hash) -> uint8_t {
  // Mask one over the high bit so that engaged control bytes are easily
  // identified.
  return (hash >> (sizeof(hash) * 8 - 7)) | 0b10000000;
}

inline auto computeHashIndex(size_t hash, int entropy) -> ssize_t {
  return hash ^ static_cast<unsigned>(entropy);
}

template <typename LookupKeyT, typename GroupT>
auto lookupIndexHashed(const LookupKeyT& lookup_key, int entropy,
                       llvm::MutableArrayRef<GroupT> groups)
    -> std::tuple<GroupT*, ssize_t> {
  //__asm volatile ("# LLVM-MCA-BEGIN hit");
  //__asm volatile ("# LLVM-MCA-BEGIN miss");
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = computeControlByte(hash);
  ssize_t hash_index = computeHashIndex(hash, entropy);

  ProbeSequence s(hash_index, groups.size());
  do {
    GroupT& g = groups[s.getIndex()];
    auto control_byte_matched_range = g.match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t byte_index = *byte_it;
        if (LLVM_LIKELY(g.Keys[byte_index] == lookup_key)) {
          return {&g, byte_index};
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots and we're done probing.
    auto empty_byte_matched_range = g.matchEmpty();
    if (LLVM_LIKELY(empty_byte_matched_range)) {
      //__asm volatile("# LLVM-MCA-END miss");
      return {nullptr, 0};
    }

    s.step();
  } while (LLVM_UNLIKELY(true));
}

template <typename KeyT, typename ValueT, typename GroupT = Group<KeyT, ValueT>>
inline auto getGroups(void* data, int size) -> llvm::MutableArrayRef<GroupT> {
  return llvm::makeMutableArrayRef(reinterpret_cast<GroupT*>(data),
                                   static_cast<unsigned>(size));
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto containsHashed(const LookupKeyT& lookup_key, int size, int entropy,
                    void* data) -> bool {
  auto groups = getGroups<KeyT, ValueT>(data, size);
  return std::get<0>(lookupIndexHashed(lookup_key, entropy, groups)) != nullptr;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookupHashed(const LookupKeyT& lookup_key, int size, int entropy,
                  void* data) -> LookupKVResult<KeyT, ValueT> {
  using GroupT = Group<KeyT, ValueT>;
  llvm::MutableArrayRef<GroupT> groups = getGroups<KeyT, ValueT>(data, size);

  GroupT* g;
  ssize_t byte_index;
  std::tie(g, byte_index) = lookupIndexHashed(lookup_key, entropy, groups);
  if (!g) {
    return {nullptr, nullptr};
  }

  return {&g->Keys[byte_index], &g->Values[byte_index]};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto contains(const LookupKeyT& lookup_key, int size, int entropy,
              int small_size, void* data) -> bool {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && entropy < 0) {
    return containsSmallLinear<KeyT>(lookup_key, size, data);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return containsHashed<KeyT, ValueT>(lookup_key, size, entropy, data);
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookup(const LookupKeyT& lookup_key, int size, int entropy, int small_size,
            void* data) -> LookupKVResult<KeyT, ValueT> {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && entropy < 0) {
    return lookupSmallLinear<KeyT, ValueT>(lookup_key, size, small_size, data);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return lookupHashed<KeyT, ValueT>(lookup_key, size, entropy, data);
}

template <typename KeyT, typename ValueT, typename CallbackT>
void forEachLinear(ssize_t size, int small_size, void* data,
                   CallbackT callback) {
  KeyT* keys = getLinearKeys<KeyT>(data);
  ValueT* values = getLinearValues<KeyT, ValueT>(data, small_size);
  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    callback(keys[i], values[i]);
  }
}

template <typename KeyT, typename ValueT, typename KVCallbackT,
          typename GroupCallbackT>
void forEachHashed(ssize_t size, void* data, KVCallbackT kv_callback,
                   GroupCallbackT group_callback) {
  using GroupT = Group<KeyT, ValueT>;
  llvm::MutableArrayRef<GroupT> groups = getGroups<KeyT, ValueT>(data, size);
  for (GroupT& g : groups) {
    auto present_matched_range = g.matchPresent();
    if (!present_matched_range) {
      continue;
    }
    for (ssize_t byte_index : present_matched_range) {
      kv_callback(g.Keys[byte_index], g.Values[byte_index]);
    }

    group_callback(g);
  }
}

template <typename KeyT, typename ValueT, typename CallbackT>
void forEach(int input_size, int entropy, int small_size, void* data,
             CallbackT callback) {
  // We manually zero-extend size so that we can use simpler signed arithmetic
  // but not pay for a sign extension when doing pointer arithmetic.
  ssize_t size = static_cast<unsigned>(input_size);

  if (shouldUseLinearLookup<KeyT>(small_size) && entropy < 0) {
    forEachLinear<KeyT, ValueT>(size, small_size, data, callback);
    return;
  }

  // Otherwise walk the non-empty slots in each control group.
  forEachHashed<KeyT, ValueT>(size, data, callback,
                              [](Group<KeyT, ValueT>& /*group*/) {});
}

template <typename GroupT>
class InsertIndexHashedResult {
 public:
  InsertIndexHashedResult(GroupT* g, bool needs_insertion, ssize_t byte_index)
      : GroupAndNeedsInsertion(g, needs_insertion), ByteIndex(byte_index) {}

  auto getGroup() -> GroupT* { return GroupAndNeedsInsertion.getPointer(); }
  auto needsInsertion() -> bool { return GroupAndNeedsInsertion.getInt(); }
  auto getByteIndex() -> ssize_t { return ByteIndex; }

 private:
  llvm::PointerIntPair<GroupT*, 1, bool> GroupAndNeedsInsertion;
  ssize_t ByteIndex;
};

// Tries to insert the given lookup key into the map. Returns three pieces of
// data compressed into two registers (in order to avoid an in-memory return).
// These are the group pointer, a bool representing whether insertion is in fact
// required, and the byte index of either the found entry in the group or the
// slot of the group to insert into. The group pointer will be null if insertion
// isn't possible without growing.
template <typename LookupKeyT, typename GroupT>
auto insertIndexHashed(const LookupKeyT& lookup_key, int growth_budget,
                       int entropy, llvm::MutableArrayRef<GroupT> groups)
    -> InsertIndexHashedResult<GroupT> {
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = computeControlByte(hash);
  ssize_t hash_index = computeHashIndex(hash, entropy);

  GroupT* group_with_deleted = nullptr;
  typename GroupT::template MatchedByteRange<> deleted_matched_range;

  auto return_insert_at_index =
      [&](GroupT& g, ssize_t byte_index) -> InsertIndexHashedResult<GroupT> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    g.setByte(byte_index, control_byte);
    return {&g, /*NeedsInsertion*/ true, byte_index};
  };

  for (ProbeSequence s(hash_index, groups.size());; s.step()) {
    auto& g = groups[s.getIndex()];

    auto control_byte_matched_range = g.match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t byte_index = *byte_it;
        if (LLVM_LIKELY(g.Keys[byte_index] == lookup_key)) {
          return {&g, /*NeedsInsertion=*/false, byte_index};
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // Track the first group with a deleted entry that we could insert over.
    if (!group_with_deleted) {
      deleted_matched_range = g.matchDeleted();
      if (deleted_matched_range) {
        group_with_deleted = &g;
      }
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // no empty slots. In that case, we'll continue probing.
    auto empty_matched_range = g.matchEmpty();
    if (!empty_matched_range) {
      continue;
    }

    // Ok, we've finished probing without finding anything and need to insert
    // instead.
    if (group_with_deleted) {
      // If we found a deleted slot, we don't need the probe sequence to insert
      // so just bail.
      break;
    }

    // Otherwise, we're going to need to grow by inserting over one of these
    // empty slots. Check that we have the budget for that before we compute the
    // exact index of the empty slot. Without the growth budget we'll have to
    // completely rehash and so we can just bail here.
    if (growth_budget == 0) {
      // Without room to grow, return that no group is viable but also set the
      // index to be negative. This ensures that a positive index is always
      // sufficient to determine that an existing was found.
      return {nullptr, /*NeedsInsertion*/ true, 0};
    }

    return return_insert_at_index(g, *empty_matched_range.begin());
  }

  return return_insert_at_index(*group_with_deleted,
                                *deleted_matched_range.begin());
}

template <typename LookupKeyT, typename GroupT>
auto insertIntoEmptyIndex(const LookupKeyT& lookup_key, int entropy,
                          llvm::MutableArrayRef<GroupT> groups)
    -> std::pair<GroupT*, ssize_t> {
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = computeControlByte(hash);
  ssize_t hash_index = computeHashIndex(hash, entropy);

  for (ProbeSequence s(hash_index, groups.size());; s.step()) {
    auto& g = groups[s.getIndex()];

    if (auto empty_matched_range = g.matchEmpty()) {
      ssize_t first_byte_index = *empty_matched_range.begin();
      g.setByte(first_byte_index, control_byte);
      return {&g, first_byte_index};
    }

    // Otherwise we continue probing.
  }
}

inline auto computeNewSize(int old_size, bool is_old_linear) -> int {
  // If the old size is a linear size, scale it by the group size.
  if (is_old_linear) {
    old_size /= GroupSize;
  }

  if (old_size < 4) {
    // If we're going to heap allocate, get at least four groups.
    return 4;
  }

  // Otherwise, we want the next power of two. This should always be a power of
  // two coming in, and so we just verify that. Also verify that this doesn't
  // overflow.
  assert(old_size == (int)llvm::PowerOf2Ceil(old_size) &&
         "Expected a power of two!");
  return old_size * 2;
}

inline auto growthThresholdForSize(int size) -> int {
  // We use a 7/8ths load factor to trigger growth.
  size *= GroupSize;
  return size - size / 8;
}

template <typename KeyT, typename ValueT, typename GroupT = Group<KeyT, ValueT>>
auto growAndRehash(MapInfo& info, int small_size, void* data,
                   GroupT*& groups_ptr) -> llvm::MutableArrayRef<GroupT> {
  // Capture the old values we will use early to make in unambiguous that they
  // aren't being updated.
  int old_size = info.Size;
  int old_entropy = info.Entropy;
  void* old_data = data;

  bool is_old_linear =
      shouldUseLinearLookup<KeyT>(small_size) && old_entropy < 0;

  // Build up the new structure after growth without modifying the underlying
  // structures. This is important as some of those may be aliased by a small
  // buffer. We don't want to change the underlying state until we copy
  // everything out.
  int new_size = computeNewSize(old_size, is_old_linear);
  auto* new_groups_ptr = new GroupT[new_size]();
  llvm::MutableArrayRef<GroupT> new_groups(new_groups_ptr, new_size);
  int new_growth_budget = growthThresholdForSize(new_size);
  int new_entropy =
      llvm::hash_combine(old_entropy, new_groups_ptr) & EntropyMask;

  forEach<KeyT, ValueT>(
      old_size, old_entropy, small_size, old_data,
      [&](KeyT& old_key, ValueT& old_value) {
        GroupT* g;
        ssize_t byte_index;
        std::tie(g, byte_index) =
            insertIntoEmptyIndex(old_key, new_entropy, new_groups);

        --new_growth_budget;
        new (&g->Keys[byte_index]) KeyT(std::move(old_key));
        old_key.~KeyT();
        new (&g->Values[byte_index]) ValueT(std::move(old_value));
        old_value.~ValueT();
      });
  assert(new_growth_budget >= 0 &&
         "Must still have a growth budget after rehash!");

  if (old_entropy >= 0) {
    // Old groups isn't a small buffer, so we need to deallocate it.
    delete[] getGroups<KeyT, ValueT>(old_data, old_size).data();
  }

  // Now that we've fully built the new, grown structures, replace the entries
  // in the data structure. At this point we can be certain to not clobber
  // anything aliasing a small buffer.
  info.Size = new_size;
  info.GrowthBudget = new_growth_budget;
  info.Entropy = new_entropy;
  groups_ptr = new_groups_ptr;

  // We return the newly allocated groups for immediate use by the caller.
  return new_groups;
}

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT = Group<KeyT, ValueT>>
auto insertHashed(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    MapInfo& info, int small_size, void* data, GroupT*& groups_ptr)
    -> InsertKVResult<KeyT, ValueT> {
  int& size = info.Size;
  int& growth_budget = info.GrowthBudget;
  int& entropy = info.Entropy;
  llvm::MutableArrayRef<GroupT> groups = getGroups<KeyT, ValueT>(data, size);

  auto result = insertIndexHashed(lookup_key, growth_budget, entropy, groups);
  GroupT* g = result.getGroup();
  ssize_t byte_index = result.getByteIndex();
  if (!result.needsInsertion()) {
    assert(g && "Must have a valid group when we find an existing entry.");
    return {false, g->Keys[byte_index], g->Values[byte_index]};
  }

  if (!g) {
    assert(
        growth_budget == 0 &&
        "Shouldn't need to grow the table until we exhaust our growth budget!");

    groups = growAndRehash<KeyT, ValueT>(info, small_size, data, groups_ptr);
    assert(growth_budget > 0 && "Must create growth budget after growing1");
    // Directly search for an empty index as we know we should have one.
    std::tie(g, byte_index) = insertIntoEmptyIndex(lookup_key, entropy, groups);
  }

  assert(g && "Should have a group to insert into now.");
  assert(byte_index >= 0 && "Should have a positive byte index now.");
  assert(growth_budget >= 0 && "Cannot insert with zero budget!");
  --growth_budget;

  KeyT* k;
  ValueT* v;
  std::tie(k, v) =
      insert_cb(lookup_key, &g->Keys[byte_index], &g->Values[byte_index]);
  return {true, *k, *v};
}

template <typename KeyT, typename ValueT, typename LookupKeyT,
          typename GroupT = Group<KeyT, ValueT>>
auto insertSmallLinear(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    MapInfo& info, int small_size, void* data, GroupT*& groups_ptr)
    -> InsertKVResult<KeyT, ValueT> {
  KeyT* keys = getLinearKeys<KeyT>(data);
  ValueT* values = getLinearValues<KeyT, ValueT>(data, small_size);
  for (int i : llvm::seq(0, info.Size)) {
    if (keys[i] == lookup_key) {
      return {false, keys[i], values[i]};
    }
  }

  KeyT* k;
  ValueT* v;

  // We need to insert. First see if we have space.
  if (info.Size < small_size) {
    // We can do the easy linear insert.
    k = &keys[info.Size];
    v = &values[info.Size];
    ++info.Size;
  } else {
    // No space for a linear insert so grow into a hash table and then do
    // a hashed insert.
    llvm::MutableArrayRef<GroupT> groups =
        growAndRehash<KeyT, ValueT>(info, small_size, data, groups_ptr);
    assert(info.GrowthBudget > 0 && "Must create growth budget after growing1");
    GroupT* g;
    int byte_index;
    std::tie(g, byte_index) =
        insertIntoEmptyIndex(lookup_key, info.Entropy, groups);
    --info.GrowthBudget;
    k = &g->Keys[byte_index];
    v = &g->Values[byte_index];
  }
  std::tie(k, v) = insert_cb(lookup_key, k, v);
  return {true, *k, *v};
}

template <typename KeyT, typename ValueT, typename LookupKeyT, typename GroupT>
auto insert(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    MapInfo& info, int small_size, void* data, GroupT*& groups_ptr)
    -> InsertKVResult<KeyT, ValueT> {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && info.Entropy < 0) {
    return insertSmallLinear<KeyT, ValueT>(lookup_key, insert_cb, info,
                                           small_size, data, groups_ptr);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return insertHashed<KeyT, ValueT>(lookup_key, insert_cb, info, small_size,
                                    data, groups_ptr);
}

template <typename KeyT, typename ValueT, typename LookupKeyT, typename GroupT>
auto update(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb,
    MapInfo& info, int small_size, void* data, GroupT*& groups_ptr)
    -> InsertKVResult<KeyT, ValueT> {
  auto i_result =
      insert(lookup_key, insert_cb, info, small_size, data, groups_ptr);

  if (i_result.isInserted()) {
    return i_result;
  }

  ValueT& v = update_cb(i_result.getKey(), i_result.getValue());
  return {false, i_result.getKey(), v};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto eraseSmallLinear(const LookupKeyT& lookup_key, int& size, int small_size,
                      void* data) -> bool {
  KeyT* keys = getLinearKeys<KeyT>(data);
  ValueT* values = getLinearValues<KeyT, ValueT>(data, small_size);
  for (int i : llvm::seq(0, size)) {
    if (keys[i] == lookup_key) {
      // Found the key, so clobber this entry with the last one and decrease
      // the size by one.
      --size;
      keys[i] = std::move(keys[size]);
      keys[size].~KeyT();
      values[i] = std::move(values[size]);
      values[size].~ValueT();
      return true;
    }
  }

  return false;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto eraseHashed(const LookupKeyT& lookup_key, MapInfo& info, void* data)
    -> bool {
  using GroupT = Group<KeyT, ValueT>;
  llvm::MutableArrayRef<GroupT> groups =
      getGroups<KeyT, ValueT>(data, info.Size);

  GroupT* g;
  ssize_t byte_index;
  std::tie(g, byte_index) = lookupIndexHashed(lookup_key, info.Entropy, groups);
  if (!g) {
    return false;
  }

  g->Keys[byte_index].~KeyT();
  g->Values[byte_index].~ValueT();

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  auto empty_matched_range = g->matchEmpty();
  if (empty_matched_range) {
    g->setByte(byte_index, GroupT::Empty);
    ++info.GrowthBudget;
  } else {
    g->setByte(byte_index, GroupT::Deleted);
  }
  return true;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto erase(const LookupKeyT& lookup_key, MapInfo& info, int small_size,
           void* data) -> bool {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear erases
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && info.Entropy < 0) {
    return eraseSmallLinear<KeyT, ValueT>(lookup_key, info.Size, small_size,
                                          data);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return eraseHashed<KeyT, ValueT>(lookup_key, info, data);
}

template <typename KeyT, typename ValueT>
void clear(MapInfo& info, int small_size, void* data) {
  // We manually zero-extend size so that we can use simpler signed arithmetic
  // but not pay for a sign extension when doing pointer arithmetic.
  ssize_t size = static_cast<unsigned>(info.Size);

  auto destroy_cb = [](KeyT& k, ValueT& v) {
    // Destroy this key and value.
    k.~KeyT();
    v.~ValueT();
  };

  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && info.Entropy < 0) {
    // Destroy all the keys and values.
    forEachLinear<KeyT, ValueT>(size, small_size, data, destroy_cb);

    // Now reset the size to zero and we'll start again inserting into the
    // beginning of the small linear buffer.
    info.Size = 0;
    return;
  }

  // Otherwise walk the non-empty slots in the control group destroying each
  // one and clearing out the group.
  forEachHashed<KeyT, ValueT>(size, data, destroy_cb,
                              [](Group<KeyT, ValueT>& g) {
                                // Clear the group.
                                g.clear();
                              });

  // And reset the growth budget.
  info.GrowthBudget = growthThresholdForSize(info.Size);
}

template <typename KeyT, typename ValueT>
void init(MapInfo& info, int small_size, void* small_buffer_addr,
          void* this_addr) {
  info.Entropy = llvm::hash_value(this_addr) & EntropyMask;
  // Mark that we don't have external storage allocated.
  info.Entropy = -info.Entropy;

  if (MapInternal::shouldUseLinearLookup<KeyT>(small_size)) {
    // We use size to mean empty when doing linear lookups.
    info.Size = 0;
    // Growth budget isn't relevant as long as we're doing linear lookups.
    info.GrowthBudget = 0;
    return;
  }

  // We're not using linear lookups in the small size, so initialize it as
  // an initial hash table.
  info.Size = small_size / GroupSize;
  info.GrowthBudget = MapInternal::growthThresholdForSize(info.Size);
  auto groups = getGroups<KeyT, ValueT>(small_buffer_addr, info.Size);
  for (auto& g : groups) {
    g.clear();
  }
}

template <typename KeyT, typename ValueT, typename GroupT>
void reset(MapInfo& info, int small_size, void* small_buffer_addr,
           void* this_addr, GroupT* groups_ptr) {
  // If we have a small size and are still using it, this is just clearing.
  if (info.Entropy < 0) {
    clear<KeyT, ValueT>(info, small_size, small_buffer_addr);
    return;
  }

  // Otherwise do the first part of the clear to destroy all the elements.
  forEachHashed<KeyT, ValueT>(
      static_cast<unsigned>(info.Size), groups_ptr,
      [](KeyT& k, ValueT& v) {
        k.~KeyT();
        v.~ValueT();
      },
      [](Group<KeyT, ValueT>& /*group*/) {});

  // Deallocate the buffer.
  delete[] groups_ptr;

  // Re-initialize the whole thing.
  init<KeyT, ValueT>(info, small_size, small_buffer_addr, this_addr);
}

template <typename KeyT, typename ValueT>
void copyInit(MapInfo& info, int small_size, void* small_buffer_addr,
              void* this_addr, const MapInfo& arg_info,
              void* arg_small_buffer_addr, void* arg_addr) {}

}  // namespace MapInternal
}  // namespace Carbon

#endif
