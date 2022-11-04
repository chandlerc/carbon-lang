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
template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
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
  ssize_t size;
  int growth_budget;
  int entropy;
};

template <typename KeyT, typename ValueT>
class LookupKVResult {
 public:
  LookupKVResult() = default;
  LookupKVResult(KeyT* key, ValueT* value) : key_(key), value_(value) {}

  explicit operator bool() const { return key_ != nullptr; }

  auto getKey() const -> KeyT& { return *key_; }
  auto getValue() const -> ValueT& { return *value_; }

 private:
  KeyT* key_ = nullptr;
  ValueT* value_ = nullptr;
};

template <typename KeyT, typename ValueT>
class InsertKVResult {
 public:
  InsertKVResult() = default;
  InsertKVResult(bool inserted, KeyT& key, ValueT& value)
      : key_and_inserted_(&key, inserted), value_(&value) {}

  auto isInserted() const -> bool { return key_and_inserted_.getInt(); }

  auto getKey() const -> KeyT& { return *key_and_inserted_.getPointer(); }
  auto getValue() const -> ValueT& { return *value_; }

 private:
  llvm::PointerIntPair<KeyT*, 1, bool> key_and_inserted_;
  ValueT* value_ = nullptr;
};

// We organize the hashtable into 16-slot groups so that we can use 16 control
// bytes to understand the contents efficiently.
constexpr ssize_t GroupSize = 16;
static_assert(llvm::isPowerOf2_64(GroupSize),
              "The group size must be a constant power of two so dividing by "
              "it is a simple shift.");

struct Group;
template <bool IsCmpVec = true>
class GroupMatchedByteRange;

class GroupMatchedByteIterator
    : public llvm::iterator_facade_base<GroupMatchedByteIterator,
                                        std::forward_iterator_tag, ssize_t, ssize_t> {
  friend struct Group;
  friend class GroupMatchedByteRange</*IsCmpVec=*/true>;
  friend class GroupMatchedByteRange</*IsCmpVec=*/false>;

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

template <bool IsCmpVec>
class GroupMatchedByteRange {
  friend struct Group;
  using MatchedByteIterator = GroupMatchedByteIterator;

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
  using MatchedByteRange = GroupMatchedByteRange<IsCmpVec>;

#if CARBON_USE_SSE_CONTROL_GROUP
  __m128i byte_vec = {};
#else
  alignas(GroupSize) std::array<uint8_t, GroupSize> Bytes = {};
#endif

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
    for (ssize_t byte_index : llvm::seq(0, GroupSize)) {
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
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, GroupSize)) {
      mask |= static_cast<bool>(Bytes[byte_index] & 0b10000000U) << byte_index;
    }
    return MatchedByteRange</*IsCmpVec=*/false>(mask);
#endif
  }

  void setByte(ssize_t byte_index, uint8_t control_byte) {
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
    for (ssize_t byte_index : llvm::seq(0, GroupSize)) {
      Bytes[byte_index] = Empty;
    }
#endif
  }
};

// We use pointers to this empty class to model the pointer to a dynamically
// allocated structure of arrays with the groups, keys, and values.
//
// This also lets us define statically allocated storage as subclasses.
struct Storage {};

template <typename KeyT, typename ValueT>
constexpr ssize_t StorageAlignment =
    std::max<ssize_t>({GroupSize, alignof(Group), alignof(KeyT), alignof(ValueT)});

template <typename KeyT>
constexpr auto ComputeKeyStorageOffset(ssize_t size) -> ssize_t {
  // Skip the group bytes.
  ssize_t offset = llvm::alignTo<GroupSize>(size);

  // And any alignment needed for the key type.
  return llvm::alignTo<alignof(KeyT)>(offset);
}

template <typename KeyT, typename ValueT>
constexpr auto ComputeValueStorageOffset(ssize_t size) -> ssize_t {
  // Skip the keys themselves.
  ssize_t offset = sizeof(KeyT) * size;

  // And skip the alignment for the value type.
  return llvm::alignTo<alignof(ValueT)>(offset);
}

template <typename KeyT, typename ValueT>
constexpr auto ComputeStorageSize(ssize_t size) -> ssize_t {
  return ComputeKeyStorageOffset<KeyT>(size) +
         ComputeValueStorageOffset<KeyT, ValueT>(size) + sizeof(ValueT) * size;
}

constexpr int EntropyMask = INT_MAX & ~(GroupSize - 1);

template <typename KeyT>
constexpr auto getKeysInCacheline() -> ssize_t {
  constexpr ssize_t CachelineSize = 64;

  return CachelineSize / sizeof(KeyT);
}

template <typename KeyT>
constexpr auto shouldUseLinearLookup(ssize_t small_size) -> bool {
  return small_size <= getKeysInCacheline<KeyT>();
}

template <typename KeyT, typename ValueT, bool UseLinearLookup, ssize_t SmallSize>
struct SmallSizeStorage;

template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT, ValueT>)
    SmallSizeStorage<KeyT, ValueT, true, SmallSize> : Storage {
  union {
    KeyT keys[SmallSize];
  };
  union {
    ValueT values[SmallSize];
  };
};

template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT, ValueT>)
    SmallSizeStorage<KeyT, ValueT, false, SmallSize> : Storage {
  // FIXME: One interesting question is whether the small size should be a
  // minumum here or an exact figure.
  static_assert(llvm::isPowerOf2_64(SmallSize),
                "SmallSize must be a power of two for a hashed buffer!");
  static_assert(SmallSize >= MapInternal::GroupSize,
                "SmallSize must be at least the size of one group!");
  static_assert((SmallSize % MapInternal::GroupSize) == 0,
                "SmallSize must be a multiple of the group size!");
  static constexpr ssize_t SmallNumGroups = SmallSize / MapInternal::GroupSize;
  static_assert(llvm::isPowerOf2_64(SmallNumGroups),
                "The number of groups must be a power of two when hashing!");

  MapInternal::Group groups[SmallNumGroups];

  union {
    KeyT keys[SmallSize];
  };
  union {
    ValueT values[SmallSize];
  };
};

template <typename KeyT, typename ValueT, bool UseLinearLookup, ssize_t SmallSize>
union StorageImpl {
  MapInternal::Storage* allocated_storage_;
  mutable MapInternal::SmallSizeStorage<KeyT, ValueT, UseLinearLookup, SmallSize>
      small_storage_;
};

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto contains(const LookupKeyT& lookup_key, ssize_t size, int entropy,
              ssize_t small_size, Storage* storage) -> bool;

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookup(const LookupKeyT& lookup_key, ssize_t size, int entropy, ssize_t small_size,
            Storage* storage) -> LookupKVResult<KeyT, ValueT>;

template <typename KeyT, typename ValueT, typename CallbackT>
void forEach(ssize_t input_size, int entropy, ssize_t small_size, Storage* storage,
             CallbackT callback);

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto insert(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    MapInfo& info, ssize_t small_size, Storage* storage, Storage*& allocated_storage)
    -> InsertKVResult<KeyT, ValueT>;

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto update(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb,
    MapInfo& info, ssize_t small_size, Storage* storage, Storage*& allocated_storage)
    -> InsertKVResult<KeyT, ValueT>;

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto erase(const LookupKeyT& lookup_key, MapInfo& info, ssize_t small_size,
           Storage* storage) -> bool;

template <typename KeyT, typename ValueT>
void clear(MapInfo& info, ssize_t small_size, Storage* storage);

template <typename KeyT, typename ValueT>
void init(MapInfo& info, ssize_t small_size, Storage* small_storage,
          void* this_addr);

template <typename KeyT, typename ValueT>
void reset(MapInfo& info, ssize_t small_size, Storage* small_storage,
           void* this_addr, Storage* storage);

template <typename KeyT, typename ValueT>
void destroy(MapInfo& info, ssize_t small_size, Storage* storage);

}  // namespace MapInternal

template <typename InputKeyT, typename InputValueT>
class MapView {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using LookupKVResultT = typename MapInternal::LookupKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  auto contains(const LookupKeyT& lookup_key) const -> bool {
    return MapInternal::contains<KeyT, ValueT>(lookup_key, size_, entropy_,
                                               small_size(), storage_);
  }

  template <typename LookupKeyT>
  auto lookup(const LookupKeyT& lookup_key) const -> LookupKVResultT {
    return MapInternal::lookup<KeyT, ValueT>(lookup_key, size_, entropy_,
                                             small_size(), storage_);
  }

  template <typename LookupKeyT>
  auto operator[](const LookupKeyT& lookup_key) const -> ValueT* {
    auto result = lookup(lookup_key);
    return result ? &result.getValue() : nullptr;
  }

  template <typename CallbackT>
  void forEach(CallbackT callback) {
    MapInternal::forEach<KeyT, ValueT>(size_, entropy_, small_size(), storage_,
                                       callback);
  }

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize>
  friend class Map;
  friend class MapRef<KeyT, ValueT>;

  MapView(ssize_t size, int entropy, ssize_t small_size, MapInternal::Storage* storage)
      : size_(size), entropy_(entropy), small_size_(small_size), storage_(storage) {}

  ssize_t size_;
  int entropy_;
  int small_size_;
  MapInternal::Storage* storage_;

  template <typename LookupKeyT>
  static auto contains(const LookupKeyT& lookup_key, ssize_t size, ssize_t entropy,
                       ssize_t small_size, void* data) -> bool;

  template <typename LookupKeyT>
  static auto lookup(const LookupKeyT& lookup_key, ssize_t size, ssize_t entropy,
                     ssize_t small_size, void* data) -> LookupKVResultT;

  auto small_size() const -> ssize_t {
    // Force zero-extend this as we store it in a smaller type.
    return static_cast<unsigned>(small_size_);
  }
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
        lookup_key, size(), entropy(), small_size_, storage());
  }

  template <typename LookupKeyT>
  auto lookup(const LookupKeyT& lookup_key) const -> LookupKVResultT {
    return MapInternal::lookup<KeyT, ValueT>(
        lookup_key, size(), entropy(), small_size_, storage());
  }

  template <typename LookupKeyT>
  auto operator[](const LookupKeyT& lookup_key) const -> ValueT* {
    auto result = lookup(lookup_key);
    return result ? &result.getValue() : nullptr;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const {
    return {size(), entropy(), small_size_, storage()};
  }

  template <typename LookupKeyT>
  auto insert(
      const LookupKeyT& lookup_key,
      typename std::__type_identity<llvm::function_ref<std::pair<
          KeyT*, ValueT*>(const LookupKeyT& lookup_key, void* key_storage,
                          void* value_storage)>>::type insert_cb)
      -> InsertKVResultT {
    return MapInternal::insert<KeyT, ValueT, LookupKeyT>(
        lookup_key, insert_cb, *info_, small_size_, storage(), *allocated_storage_ptr_);
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
                                             *info_, small_size_, storage(),
                                             *allocated_storage_ptr_);
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
    return MapInternal::erase<KeyT, ValueT>(lookup_key, *info_, small_size_,
                                            storage());
  }

  template <typename CallbackT>
  void forEach(CallbackT callback) {
    MapInternal::forEach<KeyT, ValueT>(size(), entropy(), small_size_,
                                       storage(), callback);
  }

  void clear() {
    MapInternal::clear<KeyT, ValueT>(*info_, small_size_, storage());
  }

  void reset() {
    MapInternal::reset<KeyT, ValueT>(*info_, small_size_, small_storage_, info_, *allocated_storage_ptr_);
  }

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize>
  friend class Map;

  MapRef(MapInternal::MapInfo& info, ssize_t small_size,
         MapInternal::Storage& small_storage,
         MapInternal::Storage*& allocated_storage_ptr)
      : info_(&info), small_size_(small_size) {
    if (!is_small()) {
      allocated_storage_ptr_ = &allocated_storage_ptr;
    } else {
      small_storage_ = &small_storage;
    }
  }

  MapInternal::MapInfo* info_;
  ssize_t small_size_;
  union {
    MapInternal::Storage** allocated_storage_ptr_;
    MapInternal::Storage* small_storage_;
  };

  auto size() const -> ssize_t { return info_->size; }
  auto growth_budget() const -> int { return info_->growth_budget; }
  auto entropy() const -> int { return info_->entropy; }

  auto storage() const -> MapInternal::Storage* {
    return !is_small() ? *allocated_storage_ptr_
                      : small_storage_;
  }

  auto is_small() const -> bool { return entropy() < 0; }
};

template <typename InputKeyT, typename InputValueT, ssize_t MinSmallSize = 0>
class Map {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using RefT = MapRef<KeyT, ValueT>;
  using LookupKVResultT = MapInternal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = MapInternal::InsertKVResult<KeyT, ValueT>;

  Map() {
    MapInternal::init<KeyT, ValueT>(info_, SmallSize, small_storage(), &info_);
  }
  ~Map() {
    MapInternal::destroy<KeyT, ValueT>(info_, SmallSize, storage());
  }
  Map(const Map& arg) : Map() {
    arg.forEach([this](KeyT& k, ValueT& v) { insert(k, v); });
  }
  template <ssize_t OtherMinSmallSize>
  explicit Map(const Map<KeyT, ValueT, OtherMinSmallSize>& arg) : Map() {
    arg.forEach([this](KeyT& k, ValueT& v) { insert(k, v); });
  }
  Map(Map&& arg) = delete;

  template <typename LookupKeyT>
  auto contains(const LookupKeyT& lookup_key) const -> bool {
    return MapInternal::contains<KeyT, ValueT>(
        lookup_key, size(), entropy(), SmallSize, storage());
  }

  template <typename LookupKeyT>
  auto lookup(const LookupKeyT& lookup_key) const -> LookupKVResultT {
    return MapInternal::lookup<KeyT, ValueT>(
        lookup_key, size(), entropy(), SmallSize, storage());
  }

  template <typename LookupKeyT>
  auto operator[](const LookupKeyT& lookup_key) const -> ValueT* {
    auto result = lookup(lookup_key);
    return result ? &result.getValue() : nullptr;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const {
    return {size(), entropy(), SmallSize, storage()};
  }

  template <typename LookupKeyT>
  auto insert(
      const LookupKeyT& lookup_key,
      typename std::__type_identity<llvm::function_ref<std::pair<
          KeyT*, ValueT*>(const LookupKeyT& lookup_key, void* key_storage,
                          void* value_storage)>>::type insert_cb)
      -> InsertKVResultT {
    return MapInternal::insert<KeyT, ValueT, LookupKeyT>(
        lookup_key, insert_cb, info_, SmallSize, storage(), allocated_storage_);
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
                                             info_, SmallSize, storage(),
                                             allocated_storage_);
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
    return MapInternal::erase<KeyT, ValueT>(lookup_key, info_, SmallSize,
                                            storage());
  }

  template <typename CallbackT>
  void forEach(CallbackT callback) {
    MapInternal::forEach<KeyT, ValueT>(size(), entropy(), SmallSize,
                                       storage(), callback);
  }

  void clear() {
    MapInternal::clear<KeyT, ValueT>(info_, SmallSize, storage());
  }

  void reset() {
    MapInternal::reset<KeyT, ValueT>(info_, SmallSize, small_storage(), &info_,
                                     allocated_storage_);
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator RefT() {
    return {info_, SmallSize, *small_storage(), allocated_storage_};
  }

 private:
  static constexpr ssize_t LinearSizeInPointer =
      sizeof(void*) / (sizeof(KeyT) + sizeof(ValueT));
  static constexpr ssize_t SmallSize =
      MinSmallSize < LinearSizeInPointer ? LinearSizeInPointer : MinSmallSize;
  static constexpr bool UseLinearLookup =
      MapInternal::shouldUseLinearLookup<KeyT>(SmallSize);

  static_assert(SmallSize >= 0, "Cannot have a negative small size!");

  using SmallSizeBufferT = MapInternal::SmallSizeStorage<KeyT, ValueT, UseLinearLookup, SmallSize>;

  // Validate a collection of invariants between the small size storage layout
  // and the dynamically computed storage layout. We need to do this after both
  // are complete but in the context of a specific key type, value type, and
  // small size, so here is the best place.
  static_assert(SmallSize == 0 || (alignof(SmallSizeBufferT) ==
                                   MapInternal::StorageAlignment<KeyT, ValueT>),
                "Small size buffer must have the same alignment as a heap "
                "allocated buffer.");
  static_assert(
      SmallSize == 0 ||
          (offsetof(SmallSizeBufferT, keys) ==
           (UseLinearLookup
                ? 0
                : MapInternal::ComputeKeyStorageOffset<KeyT>(SmallSize))),
      "Offset to keys in small size storage doesn't match computed offset!");
  static_assert(
      SmallSize == 0 ||
          (offsetof(SmallSizeBufferT, values) ==
           (UseLinearLookup
                ? 0
                : MapInternal::ComputeKeyStorageOffset<KeyT>(SmallSize)) +
               MapInternal::ComputeValueStorageOffset<KeyT, ValueT>(SmallSize)),
      "Offset from keys to values in small size storage doesn't match computed "
      "offset!");
  static_assert(SmallSize == 0 || UseLinearLookup ||
                    (sizeof(SmallSizeBufferT) ==
                     MapInternal::ComputeStorageSize<KeyT, ValueT>(SmallSize)),
                "The small size storage needs to match the dynamically "
                "computed storage size.");

  MapInternal::MapInfo info_;

  union {
    MapInternal::Storage* allocated_storage_;
    mutable MapInternal::SmallSizeStorage<KeyT, ValueT, UseLinearLookup,
                                         SmallSize>
        small_storage_;
  };

  auto size() const -> ssize_t { return info_.size; }
  auto growth_budget() const -> int { return info_.growth_budget; }
  auto entropy() const -> int { return info_.entropy; }

  auto is_small() const -> bool { return entropy() < 0; }

  auto small_storage() const -> MapInternal::Storage* {
    return &small_storage_;
  }

  auto storage() const -> MapInternal::Storage* {
    return !is_small() ? allocated_storage_ : small_storage();
  }

};

// Implementation of the routines in `map_internal` that are used above.
namespace MapInternal {

template <typename KeyT>
auto getLinearKeys(Storage* storage) -> KeyT* {
  return reinterpret_cast<KeyT*>(storage);
}

template <typename KeyT, typename LookupKeyT>
auto containsSmallLinear(const LookupKeyT& lookup_key, ssize_t size, Storage* storage)
    -> bool {
  KeyT* keys = getLinearKeys<KeyT>(storage);
  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    if (keys[i] == lookup_key) {
      return true;
    }
  }

  return false;
}

template <typename KeyT, typename ValueT>
auto getLinearValues(Storage* storage, ssize_t small_size) -> ValueT* {
  return reinterpret_cast<ValueT*>(
      storage + ComputeValueStorageOffset<KeyT, ValueT>(small_size));
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookupSmallLinear(const LookupKeyT& lookup_key, ssize_t size, ssize_t small_size,
                       Storage* storage) -> LookupKVResult<KeyT, ValueT> {
  KeyT* keys = getLinearKeys<KeyT>(storage);
  ValueT* values = getLinearValues<KeyT, ValueT>(storage, small_size);
  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    if (keys[i] == lookup_key) {
      return {&keys[i], &values[i]};
    }
  }

  return {nullptr, nullptr};
}

inline auto computeProbeMaskFromSize(ssize_t groups_size) -> size_t {
  assert(llvm::isPowerOf2_64(groups_size) &&
         "Size must be a power of two for a hashed buffer!");
  // The probe mask needs to mask down to keep the index within
  // `groups_size`. Since `groups_size` is a power of two, this is equivalent to
  // `groups_size - 1`.
  return groups_size - 1;
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

template <typename LookupKeyT, typename KeyT>
auto lookupIndexHashed(const LookupKeyT& lookup_key, int entropy,
                       llvm::MutableArrayRef<Group> groups, KeyT* keys)
    -> ssize_t {
  //__asm volatile ("# LLVM-MCA-BEGIN hit");
  //__asm volatile ("# LLVM-MCA-BEGIN miss");
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = computeControlByte(hash);
  ssize_t hash_index = computeHashIndex(hash, entropy);

  ProbeSequence s(hash_index, groups.size());
  do {
    ssize_t group_index = s.getIndex();
    Group& g = groups[group_index];
    auto control_byte_matched_range = g.match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = group_index * GroupSize + *byte_it;
        if (LLVM_LIKELY(keys[index] == lookup_key)) {
          return index;
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots and we're done probing.
    auto empty_byte_matched_range = g.matchEmpty();
    if (LLVM_LIKELY(empty_byte_matched_range)) {
      //__asm volatile("# LLVM-MCA-END miss");
      return -1;
    }

    s.step();
  } while (LLVM_UNLIKELY(true));
}

inline auto getGroupsSize(ssize_t size) -> ssize_t {
  return size / GroupSize;
}

inline auto getRawStorage(Storage* storage) -> unsigned char * {
  return reinterpret_cast<unsigned char*>(storage);
}

inline auto getGroups(Storage* storage, ssize_t size) -> llvm::MutableArrayRef<Group> {
  return llvm::makeMutableArrayRef(reinterpret_cast<Group*>(storage),
                                   getGroupsSize(size));
}

template <typename KeyT>
inline auto getKeys(Storage* storage, ssize_t size) -> KeyT* {
  return reinterpret_cast<KeyT*>(getRawStorage(storage) + ComputeKeyStorageOffset<KeyT>(size));
}

template <typename KeyT, typename ValueT>
inline auto getValues(Storage* storage, ssize_t size) -> ValueT* {
  return reinterpret_cast<ValueT*>(getRawStorage(storage) + ComputeKeyStorageOffset<KeyT>(size) +
                                   ComputeValueStorageOffset<KeyT, ValueT>(size));
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto containsHashed(const LookupKeyT& lookup_key, ssize_t size, int entropy,
                    Storage* storage) -> bool {
  llvm::MutableArrayRef<Group> groups = getGroups(storage, size);
  KeyT* keys = getKeys<KeyT>(storage, size);
  return lookupIndexHashed(lookup_key, entropy, groups, keys) >= 0;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookupHashed(const LookupKeyT& lookup_key, ssize_t size, int entropy,
                  Storage* storage) -> LookupKVResult<KeyT, ValueT> {
  llvm::MutableArrayRef<Group> groups = getGroups(storage, size);
  KeyT* keys = getKeys<KeyT>(storage, size);
  ValueT* values = getValues<KeyT, ValueT>(storage, size);

  ssize_t index = lookupIndexHashed(lookup_key, entropy, groups, keys);
  if (index < 0) {
    return {nullptr, nullptr};
  }

  return {&keys[index], &values[index]};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto contains(const LookupKeyT& lookup_key, ssize_t size, int entropy,
              ssize_t small_size, Storage* storage) -> bool {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && entropy < 0) {
    return containsSmallLinear<KeyT>(lookup_key, size, storage);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return containsHashed<KeyT, ValueT>(lookup_key, size, entropy, storage);
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookup(const LookupKeyT& lookup_key, ssize_t size, int entropy, ssize_t small_size,
            Storage* storage) -> LookupKVResult<KeyT, ValueT> {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && entropy < 0) {
    return lookupSmallLinear<KeyT, ValueT>(lookup_key, size, small_size, storage);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return lookupHashed<KeyT, ValueT>(lookup_key, size, entropy, storage);
}

template <typename KeyT, typename ValueT, typename CallbackT>
void forEachLinear(ssize_t size, ssize_t small_size, Storage* storage,
                   CallbackT callback) {
  KeyT* keys = getLinearKeys<KeyT>(storage);
  ValueT* values = getLinearValues<KeyT, ValueT>(storage, small_size);
  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    callback(keys[i], values[i]);
  }
}

template <typename KeyT, typename ValueT, typename KVCallbackT,
          typename GroupCallbackT>
void forEachHashed(ssize_t size, Storage* storage, KVCallbackT kv_callback,
                   GroupCallbackT group_callback) {
  llvm::MutableArrayRef<Group> groups = getGroups(storage, size);
  KeyT* keys = getKeys<KeyT>(storage, size);
  ValueT* values = getValues<KeyT, ValueT>(storage, size);

  for (ssize_t group_index : llvm::seq<ssize_t>(0, groups.size())) {
    Group& g = groups[group_index];
    auto present_matched_range = g.matchPresent();
    if (!present_matched_range) {
      continue;
    }
    for (ssize_t byte_index : present_matched_range) {
      ssize_t index = group_index * GroupSize + byte_index;
      kv_callback(keys[index], values[index]);
    }

    group_callback(g);
  }
}

template <typename KeyT, typename ValueT, typename CallbackT>
void forEach(ssize_t size, int entropy, ssize_t small_size, Storage* storage,
             CallbackT callback) {
  if (shouldUseLinearLookup<KeyT>(small_size) && entropy < 0) {
    forEachLinear<KeyT, ValueT>(size, small_size, storage, callback);
    return;
  }

  // Otherwise walk the non-empty slots in each control group.
  forEachHashed<KeyT, ValueT>(size, storage, callback, [](Group& /*group*/) {});
}

// Tries to insert the given lookup key into the map. Returns three pieces of
// data compressed into two registers (in order to avoid an in-memory return).
// These are the group pointer, a bool representing whether insertion is in fact
// required, and the byte index of either the found entry in the group or the
// slot of the group to insert into. The group pointer will be null if insertion
// isn't possible without growing.
template <typename LookupKeyT, typename KeyT>
auto insertIndexHashed(const LookupKeyT& lookup_key, int growth_budget,
                       int entropy, llvm::MutableArrayRef<Group> groups, KeyT* keys)
    -> std::pair<bool, ssize_t> {
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = computeControlByte(hash);
  ssize_t hash_index = computeHashIndex(hash, entropy);

  ssize_t group_with_deleted_index = -1;
  Group::MatchedByteRange<> deleted_matched_range;

  auto return_insert_at_index =
      [&](ssize_t group_index, ssize_t byte_index) -> std::pair<bool, ssize_t> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    groups[group_index].setByte(byte_index, control_byte);
    return {/*needs_insertion=*/true, group_index * GroupSize + byte_index};
  };

  for (ProbeSequence s(hash_index, groups.size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto& g = groups[group_index];

    auto control_byte_matched_range = g.match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = group_index * GroupSize + *byte_it;
        if (LLVM_LIKELY(keys[index] == lookup_key)) {
          return {/*needs_insertion=*/false, index};
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // Track the first group with a deleted entry that we could insert over.
    if (group_with_deleted_index < 0) {
      deleted_matched_range = g.matchDeleted();
      if (deleted_matched_range) {
        group_with_deleted_index = group_index;
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
    if (group_with_deleted_index >= 0) {
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
      return {/*needs_insertion=*/true, -1};
    }

    return return_insert_at_index(group_index, *empty_matched_range.begin());
  }

  return return_insert_at_index(group_with_deleted_index,
                                *deleted_matched_range.begin());
}

template <typename LookupKeyT>
auto insertIntoEmptyIndex(const LookupKeyT& lookup_key, int entropy,
                          llvm::MutableArrayRef<Group> groups)
    -> ssize_t {
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = computeControlByte(hash);
  ssize_t hash_index = computeHashIndex(hash, entropy);

  for (ProbeSequence s(hash_index, groups.size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto& g = groups[group_index];

    if (auto empty_matched_range = g.matchEmpty()) {
      ssize_t first_byte_index = *empty_matched_range.begin();
      g.setByte(first_byte_index, control_byte);
      return group_index * GroupSize + first_byte_index;
    }

    // Otherwise we continue probing.
  }
}

template <typename KeyT, typename ValueT>
auto allocateStorage(ssize_t size) -> Storage* {
  ssize_t allocated_size = ComputeStorageSize<KeyT, ValueT>(size);
  return reinterpret_cast<Storage*>(__builtin_operator_new(
      allocated_size, std::align_val_t(StorageAlignment<KeyT, ValueT>),
      std::nothrow_t()));
}

template <typename KeyT, typename ValueT>
void deallocateStorage(Storage* storage, ssize_t size) {
#if __cpp_sized_deallocation
  ssize_t allocated_size = computeStorageSize<KeyT, ValueT>(size);
  return __builtin_operator_delete(
      storage, allocated_size,
      std::align_val_t(StorageAlignment<KeyT, ValueT>));
#else
  // Ensure `size` is used even in the fallback non-sized deallocation case.
  (void)size;
  return __builtin_operator_delete(
      storage, std::align_val_t(StorageAlignment<KeyT, ValueT>));
#endif
}

inline auto computeNewSize(ssize_t old_size) -> ssize_t {
  if (old_size < (4 * GroupSize)) {
    // If we're going to heap allocate, get at least four groups.
    return 4 * GroupSize;
  }

  // Otherwise, we want the next power of two. This should always be a power of
  // two coming in, and so we just verify that. Also verify that this doesn't
  // overflow.
  assert(old_size == (ssize_t)llvm::PowerOf2Ceil(old_size) &&
         "Expected a power of two!");
  return old_size * 2;
}

inline auto growthThresholdForSize(ssize_t size) -> ssize_t {
  // We use a 7/8ths load factor to trigger growth.
  return size - size / 8;
}

template <typename KeyT, typename ValueT>
auto growAndRehash(MapInfo& info, ssize_t small_size, Storage* storage, Storage*& allocated_storage)
    -> llvm::MutableArrayRef<Group> {
  // Capture the old values we will use early to make in unambiguous that they
  // aren't being updated.
  ssize_t old_size = info.size;
  int old_entropy = info.entropy;
  Storage* old_storage = storage;

  // Build up the new structure after growth without modifying the underlying
  // structures. This is important as some of those may be aliased by a small
  // buffer. We don't want to change the underlying state until we copy
  // everything out.
  ssize_t new_size = computeNewSize(old_size);
  Storage* new_storage = allocateStorage<KeyT, ValueT>(new_size);
  llvm::MutableArrayRef<Group> new_groups = getGroups(new_storage, new_size);
  for (Group &g : new_groups) {
    g.clear();
  }
  KeyT* new_keys = getKeys<KeyT>(new_storage, new_size);
  ValueT* new_values = getValues<KeyT, ValueT>(new_storage, new_size);
  int new_growth_budget = growthThresholdForSize(new_size);
  int new_entropy =
      llvm::hash_combine(old_entropy, new_storage) & EntropyMask;

  forEach<KeyT, ValueT>(
      old_size, old_entropy, small_size, old_storage,
      [&](KeyT& old_key, ValueT& old_value) {
        ssize_t index = insertIntoEmptyIndex(old_key, new_entropy, new_groups);
        --new_growth_budget;
        new (&new_keys[index]) KeyT(std::move(old_key));
        old_key.~KeyT();
        new (&new_values[index]) ValueT(std::move(old_value));
        old_value.~ValueT();
      });
  assert(new_growth_budget >= 0 &&
         "Must still have a growth budget after rehash!");

  if (old_entropy >= 0) {
    // Old isn't a small buffer, so we need to deallocate it.
    deallocateStorage<KeyT, ValueT>(old_storage, old_size);
  }

  // Now that we've fully built the new, grown structures, replace the entries
  // in the data structure. At this point we can be certain to not clobber
  // anything aliasing a small buffer.
  info.size = new_size;
  info.growth_budget = new_growth_budget;
  info.entropy = new_entropy;
  allocated_storage = new_storage;

  // We return the newly allocated groups for immediate use by the caller.
  return new_groups;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto insertHashed(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    MapInfo& info, ssize_t small_size, Storage* storage, Storage*& allocated_storage)
    -> InsertKVResult<KeyT, ValueT> {
  ssize_t& size = info.size;
  int& growth_budget = info.growth_budget;
  int& entropy = info.entropy;
  llvm::MutableArrayRef<Group> groups = getGroups(storage, size);
  KeyT* keys = getKeys<KeyT>(storage, size);
  ValueT* values = getValues<KeyT, ValueT>(storage, size);

  auto [needs_insertion, index] =
      insertIndexHashed(lookup_key, growth_budget, entropy, groups, keys);
  if (!needs_insertion) {
    assert(index >= 0 && "Must have a valid group when we find an existing entry.");
    return {false, keys[index], values[index]};
  }

  if (index < 0) {
    assert(
        growth_budget == 0 &&
        "Shouldn't need to grow the table until we exhaust our growth budget!");

    groups = growAndRehash<KeyT, ValueT>(info, small_size, storage, allocated_storage);
    assert(growth_budget > 0 && "Must create growth budget after growing1");
    // Make sure to update our keys and values pointers based on the updated storage.
    storage = allocated_storage;
    keys = getKeys<KeyT>(storage, size);
    values = getValues<KeyT, ValueT>(storage, size);
    // Directly insert into an empty index as we know we have one.
    index = insertIntoEmptyIndex(lookup_key, entropy, groups);
  }

  assert(index >= 0 && "Should have a group to insert into now.");
  assert(growth_budget >= 0 && "Cannot insert with zero budget!");
  --growth_budget;

  KeyT* k;
  ValueT* v;
  std::tie(k, v) =
      insert_cb(lookup_key, &keys[index], &values[index]);
  return {true, *k, *v};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto insertSmallLinear(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    MapInfo& info, ssize_t small_size, Storage* storage, Storage*& allocated_storage)
    -> InsertKVResult<KeyT, ValueT> {
  KeyT* keys = getLinearKeys<KeyT>(storage);
  ValueT* values = getLinearValues<KeyT, ValueT>(storage, small_size);
  for (ssize_t i : llvm::seq<ssize_t>(0, info.size)) {
    if (keys[i] == lookup_key) {
      return {false, keys[i], values[i]};
    }
  }

  size_t index;

  // We need to insert. First see if we have space.
  if (info.size < small_size) {
    // We can do the easy linear insert.
    index = info.size;
    ++info.size;
  } else {
    // No space for a linear insert so grow into a hash table and then do
    // a hashed insert.
    llvm::MutableArrayRef<Group> groups =
        growAndRehash<KeyT, ValueT>(info, small_size, storage, allocated_storage);
    assert(info.growth_budget > 0 && "Must create growth budget after growing1");
    // Make sure to update our keys and values pointers based on the updated storage.
    storage = allocated_storage;
    keys = getKeys<KeyT>(storage, info.size);
    values = getValues<KeyT, ValueT>(storage, info.size);
    index = insertIntoEmptyIndex(lookup_key, info.entropy, groups);
    --info.growth_budget;
  }
  KeyT* k;
  ValueT* v;
  std::tie(k, v) = insert_cb(lookup_key, &keys[index], &values[index]);
  return {true, *k, *v};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto insert(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    MapInfo& info, ssize_t small_size, Storage* storage, Storage*& allocated_storage)
    -> InsertKVResult<KeyT, ValueT> {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && info.entropy < 0) {
    return insertSmallLinear<KeyT, ValueT>(lookup_key, insert_cb, info,
                                           small_size, storage, allocated_storage);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return insertHashed<KeyT, ValueT>(lookup_key, insert_cb, info, small_size,
                                    storage, allocated_storage);
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto update(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb,
    llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb,
    MapInfo& info, ssize_t small_size, Storage* storage, Storage*& allocated_storage)
    -> InsertKVResult<KeyT, ValueT> {
  auto i_result =
      insert(lookup_key, insert_cb, info, small_size, storage, allocated_storage);

  if (i_result.isInserted()) {
    return i_result;
  }

  ValueT& v = update_cb(i_result.getKey(), i_result.getValue());
  return {false, i_result.getKey(), v};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto eraseSmallLinear(const LookupKeyT& lookup_key, ssize_t& size, ssize_t small_size,
                      Storage* storage) -> bool {
  KeyT* keys = getLinearKeys<KeyT>(storage);
  ValueT* values = getLinearValues<KeyT, ValueT>(storage, small_size);
  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
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
auto eraseHashed(const LookupKeyT& lookup_key, MapInfo& info, Storage* storage)
    -> bool {
  llvm::MutableArrayRef<Group> groups = getGroups(storage, info.size);
  KeyT* keys = getKeys<KeyT>(storage, info.size);
  ValueT* values = getValues<KeyT, ValueT>(storage, info.size);

  ssize_t index = lookupIndexHashed(lookup_key, info.entropy, groups, keys);
  if (index < 0) {
    return false;
  }

  keys[index].~KeyT();
  values[index].~ValueT();

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  ssize_t group_index = index / GroupSize;
  ssize_t byte_index = index % GroupSize;
  Group &g = groups[group_index];
  auto empty_matched_range = g.matchEmpty();
  if (empty_matched_range) {
    g.setByte(byte_index, Group::Empty);
    ++info.growth_budget;
  } else {
    g.setByte(byte_index, Group::Deleted);
  }
  return true;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto erase(const LookupKeyT& lookup_key, MapInfo& info, ssize_t small_size,
           Storage* storage) -> bool {
  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear erases
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && info.entropy < 0) {
    return eraseSmallLinear<KeyT, ValueT>(lookup_key, info.size, small_size,
                                          storage);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return eraseHashed<KeyT, ValueT>(lookup_key, info, storage);
}

template <typename KeyT, typename ValueT>
void clear(MapInfo& info, ssize_t small_size, Storage* storage) {
  auto destroy_cb = [](KeyT& k, ValueT& v) {
    // Destroy this key and value.
    k.~KeyT();
    v.~ValueT();
  };

  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && info.entropy < 0) {
    // Destroy all the keys and values.
    forEachLinear<KeyT, ValueT>(info.size, small_size, storage, destroy_cb);

    // Now reset the size to zero and we'll start again inserting into the
    // beginning of the small linear buffer.
    info.size = 0;
    return;
  }

  // Otherwise walk the non-empty slots in the control group destroying each
  // one and clearing out the group.
  forEachHashed<KeyT, ValueT>(info.size, storage, destroy_cb,
                              [](Group& g) {
                                // Clear the group.
                                g.clear();
                              });

  // And reset the growth budget.
  info.growth_budget = growthThresholdForSize(info.size);
}

template <typename KeyT, typename ValueT>
void init(MapInfo& info, ssize_t small_size, Storage* small_storage,
          void* this_addr) {
  info.entropy = llvm::hash_value(this_addr) & EntropyMask;
  // Mark that we don't have external storage allocated.
  info.entropy = -info.entropy;

  if (MapInternal::shouldUseLinearLookup<KeyT>(small_size)) {
    // We use size to mean empty when doing linear lookups.
    info.size = 0;
    // Growth budget isn't relevant as long as we're doing linear lookups.
    info.growth_budget = 0;
    return;
  }

  // We're not using linear lookups in the small size, so initialize it as
  // an initial hash table.
  info.size = small_size;
  info.growth_budget = MapInternal::growthThresholdForSize(info.size);
  auto groups = getGroups(small_storage, info.size);
  for (Group& g : groups) {
    g.clear();
  }
}

template <typename KeyT, typename ValueT>
void reset(MapInfo& info, ssize_t small_size, Storage* small_storage,
           void* this_addr, Storage* storage) {
  // If we have a small size and are still using it, this is just clearing.
  if (info.entropy < 0) {
    clear<KeyT, ValueT>(info, small_size, small_storage);
    return;
  }

  // Otherwise do the first part of the clear to destroy all the elements.
  forEachHashed<KeyT, ValueT>(
      info.size, storage,
      [](KeyT& k, ValueT& v) {
        k.~KeyT();
        v.~ValueT();
      },
      [](Group& /*group*/) {});

  // Deallocate the buffer.
  deallocateStorage<KeyT, ValueT>(storage, info.size);

  // Re-initialize the whole thing.
  init<KeyT, ValueT>(info, small_size, small_storage, this_addr);
}

template <typename KeyT, typename ValueT>
void destroy(MapInfo& info, ssize_t small_size, Storage* storage) {
  auto destroy_cb = [](KeyT& k, ValueT& v) {
    // Destroy this key and value.
    k.~KeyT();
    v.~ValueT();
  };

  // The entropy will be negative when using the small buffer, and we can
  // recompute from the small buffer size whether that implies linear lookups
  // due to fitting on a cacheline.
  if (shouldUseLinearLookup<KeyT>(small_size) && info.entropy < 0) {
    // Destroy all the keys and values.
    forEachLinear<KeyT, ValueT>(info.size, small_size, storage, destroy_cb);

    // Nothing to deallocate as we're always small.
    return;
  }

  // Otherwise walk the non-empty slots in the control group destroying each
  // one. We don't need to do any clearing as we're destroying the map.
  forEachHashed<KeyT, ValueT>(info.size, storage, destroy_cb,
                              [](Group& /*group*/) {});

  // If small, nothing to deallocate.
  if (info.entropy < 0) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the object.
  deallocateStorage<KeyT, ValueT>(storage, info.size);
}

}  // namespace MapInternal
}  // namespace Carbon

#endif
