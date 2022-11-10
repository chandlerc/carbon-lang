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
class MapBase;
template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
class Map;

namespace MapInternal {

// Detect whether we can use SIMD accelerated implementations of the control
// groups.
#if 1 || defined(__SSSE3__)
#define CARBON_USE_SSE_CONTROL_GROUP 1
#if 0 && defined(__SSE4_1__)
#define CARBON_OPTIMIZE_SSE4_1 1
#endif
#endif

struct MapInfo {
  ssize_t size;
  int entropy;
  int growth_budget;
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
constexpr ssize_t GroupMask = GroupSize - 1;

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
  unsigned mask_;

  explicit GroupMatchedByteRange(unsigned mask) : mask_(mask) {}
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
    return mask_ == 0;
#endif
  }

  auto begin() const -> MatchedByteIterator {
#if CARBON_OPTIMIZE_SSE4_1
    unsigned mask = _mm_movemask_epi8(mask_vec);
    return MatchedByteIterator(mask);
#else
    return MatchedByteIterator(mask_);
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

  static auto load(uint8_t* groups, ssize_t index)
      -> Group {
    Group g;
#if CARBON_USE_SSE_CONTROL_GROUP
    g.byte_vec =
        _mm_load_si128(reinterpret_cast<__m128i*>(groups + index));
#else
    std::memcpy(&g.Bytes, groups + index, GroupSize);
#endif
    return g;
  }

  static auto load(llvm::MutableArrayRef<uint8_t> groups, ssize_t index)
      -> Group {
    Group g;
#if CARBON_USE_SSE_CONTROL_GROUP
    g.byte_vec =
        _mm_load_si128(reinterpret_cast<__m128i*>(groups.data() + index));
#else
    std::memcpy(&g.Bytes, groups.data() + index, GroupSize);
#endif
    return g;
  }

  auto match(uint8_t match_byte) const -> MatchedByteRange<> {
#if CARBON_USE_SSE_CONTROL_GROUP
    auto match_byte_vec = _mm_set1_epi8(match_byte);
    auto match_byte_cmp_vec = _mm_cmpeq_epi8(byte_vec, match_byte_vec);
#if CARBON_OPTIMIZE_SSE4_1
    return MatchedByteRange<>(match_byte_cmp_vec);
#else
    return MatchedByteRange<>((unsigned)_mm_movemask_epi8(match_byte_cmp_vec));
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
    return MatchedByteRange</*IsCmpVec=*/false>((unsigned)_mm_movemask_epi8(byte_vec));
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

#if 0
  void setByte(ssize_t byte_index, uint8_t control_byte) {
#if CARBON_USE_SSE_CONTROL_GROUP
    // We directly manipulate the storage of the vector as there isn't a nice
    // intrinsic for this.
    reinterpret_cast<unsigned char*>(&byte_vec)[byte_index] = control_byte;
#else
    Bytes[byte_index] = control_byte;
#endif
  }
#endif

#if 0
  void clear() {
#if CARBON_USE_SSE_CONTROL_GROUP
    byte_vec = _mm_set1_epi8(Empty);
#else
    for (ssize_t byte_index : llvm::seq(0, GroupSize)) {
      Bytes[byte_index] = Empty;
    }
#endif
  }
#endif
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
  // There are `size` control bytes plus any alignment needed for the key type.
  return llvm::alignTo<alignof(KeyT)>(size);
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

constexpr ssize_t CachelineSize = 64;

template <typename KeyT>
constexpr auto getKeysInCacheline() -> int {
  return CachelineSize / sizeof(KeyT);
}

template <typename KeyT, typename ValueT>
constexpr auto DefaultMinSmallSize() -> ssize_t {
  return (CachelineSize - sizeof(MapInfo)) / (sizeof(KeyT) + sizeof(ValueT));
}

template <typename KeyT>
constexpr auto ShouldUseLinearLookup(int small_size) -> bool {
  return small_size >= 0 && small_size <= getKeysInCacheline<KeyT>();
}

template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
constexpr auto ComputeSmallSize() -> ssize_t {
  constexpr ssize_t LinearSizeInPointer =
      sizeof(void*) / (sizeof(KeyT) + sizeof(ValueT));
  constexpr ssize_t SmallSizeFloor =
      MinSmallSize < LinearSizeInPointer ? LinearSizeInPointer : MinSmallSize;
  constexpr bool UseLinearLookup =
      MapInternal::ShouldUseLinearLookup<KeyT>(SmallSizeFloor);

  return UseLinearLookup
             ? SmallSizeFloor
             : llvm::alignTo<MapInternal::GroupSize>(SmallSizeFloor);
}

// Compute an artificial alignment for the map data structure to try to keep its
// small-size buffer on the same cacheline as the structure itself.
// Frustratingly we have to do this based on just the raw template parameters
// which makes it duplicate some logic from the data structure itself.
template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
constexpr auto ComputeMapAlignment() -> ssize_t {
  constexpr ssize_t SmallSize =
      MapInternal::ComputeSmallSize<KeyT, ValueT, MinSmallSize>();
  constexpr bool UseLinearLookup =
      MapInternal::ShouldUseLinearLookup<KeyT>(SmallSize);
  
  struct Baseline {
    MapInfo info;
    Storage* storage;
  };
  struct LinearLayout {
    MapInfo info;
    KeyT keys[SmallSize];
    ValueT values[SmallSize];
  };
  if (SmallSize == 0) {
    return alignof(Baseline);
  }
  if (UseLinearLookup) {
    if (sizeof(LinearLayout) <= CachelineSize) {
      // Inflate the alignment to the size when it will fit in a cacheline so we
      // don't straddle cachelines when searching in the small linear case.
      return llvm::NextPowerOf2(sizeof(LinearLayout));
    }
    return alignof(LinearLayout);
  }
  return StorageAlignment<KeyT, ValueT>;
}

template <typename KeyT, typename ValueT, bool UseLinearLookup, ssize_t SmallSize>
struct SmallSizeStorage;

template <typename KeyT, typename ValueT>
struct SmallSizeStorage<KeyT, ValueT, true, 0> : Storage {
  union {
    KeyT keys[0];
  };
  union {
    ValueT values[0];
  };
};

template <typename KeyT, typename ValueT>
struct SmallSizeStorage<KeyT, ValueT, false, 0> : Storage {
  union {
    KeyT keys[0];
  };
  union {
    ValueT values[0];
  };
};

template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct SmallSizeStorage<KeyT, ValueT, true, SmallSize> : Storage {
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

template <typename KeyT>
auto getLinearKeys(Storage* storage) -> KeyT* {
  return reinterpret_cast<KeyT*>(storage);
}

template <typename KeyT, typename ValueT>
auto getLinearValues(Storage* storage, int small_size) -> ValueT* {
  return reinterpret_cast<ValueT*>(storage +
                                   ComputeValueStorageOffset<KeyT, ValueT>(
                                       static_cast<unsigned>(small_size)));
}

inline auto getGroupsSize(ssize_t size) -> ssize_t {
  return size / GroupSize;
}

inline auto getRawStorage(Storage* storage) -> unsigned char * {
  return reinterpret_cast<unsigned char*>(storage);
}

inline auto getGroupsPtr(Storage* storage) -> uint8_t* {
  return reinterpret_cast<uint8_t*>(storage);
}

inline auto getGroups(Storage* storage, ssize_t size) -> llvm::MutableArrayRef<uint8_t> {
  assert((size % GroupSize) == 0 &&
         "Size must be an exact multiple of the group size.");
  return llvm::makeMutableArrayRef(getGroupsPtr(storage), size);
}

template <typename KeyT>
inline auto getKeys(Storage* storage, ssize_t size) -> KeyT* {
  assert(llvm::isPowerOf2_64(size) &&
         "Size must be a power of two for a hashed buffer!");
  assert(size == ComputeKeyStorageOffset<KeyT>(size) &&
         "Cannot be more aligned than a power of two.");
  return reinterpret_cast<KeyT*>(reinterpret_cast<unsigned char*>(storage) + size);
}

template <typename KeyT, typename ValueT>
inline auto getValues(Storage* storage, ssize_t size) -> ValueT* {
  return reinterpret_cast<ValueT*>(reinterpret_cast<unsigned char*>(storage) + ComputeKeyStorageOffset<KeyT>(size) +
                                   ComputeValueStorageOffset<KeyT, ValueT>(size));
}

template <typename KeyT, typename ValueT>
inline auto getValueFromKey(KeyT* key, ssize_t size) -> ValueT* {
  return reinterpret_cast<ValueT*>(
      reinterpret_cast<unsigned char*>(key) +
      ComputeValueStorageOffset<KeyT, ValueT>(size));
}

}  // namespace MapInternal

template <typename InputKeyT, typename InputValueT>
class MapView {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using LookupKVResultT = typename MapInternal::LookupKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  auto contains(const LookupKeyT& lookup_key) const -> bool;

  template <typename LookupKeyT>
  auto lookup(const LookupKeyT& lookup_key) const -> LookupKVResultT;

  template <typename LookupKeyT>
  auto operator[](const LookupKeyT& lookup_key) const -> ValueT*;

  template <typename CallbackT>
  void forEach(CallbackT callback);

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize>
  friend class Map;
  friend class MapBase<KeyT, ValueT>;

  MapView() = default;
  MapView(ssize_t size, int small_size, MapInternal::Storage* storage)
      : packed_sizes_(size | (static_cast<int64_t>(small_size) << 32)),
        storage_(storage) {
    assert(size >= 0 && "Cannot have a negative size!");
    assert(size <= INT_MAX && "Only 32-bit sizes are supported!");
  }

  int64_t packed_sizes_;
  MapInternal::Storage* storage_;

  template <typename LookupKeyT>
  static auto contains(const LookupKeyT& lookup_key, ssize_t size,
                       int small_size, void* data) -> bool;

  template <typename LookupKeyT>
  static auto lookup(const LookupKeyT& lookup_key, ssize_t size,
                     int small_size, void* data) -> LookupKVResultT;

  auto size() const -> ssize_t { return static_cast<uint32_t>(packed_sizes_); }
  auto small_size() const -> int { return packed_sizes_ >> 32; }

  void set_size(ssize_t size) {
    assert(size >= 0 && "Cannot have a negative size!");
    assert(size <= INT_MAX && "Only 32-bit sizes are supported!");
    packed_sizes_ &= -1ULL << 32;
    packed_sizes_ |= size & ((1LL << 32) - 1);
  }
  void set_small_size(int small_size) {
    packed_sizes_ &= (1ULL << 32) - 1;
    packed_sizes_ |= static_cast<int64_t>(small_size) << 32;
  }
};

template <typename InputKeyT, typename InputValueT>
class MapBase {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using LookupKVResultT = MapInternal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = MapInternal::InsertKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  auto contains(const LookupKeyT& lookup_key) const -> bool {
    return ViewT(*this).contains(lookup_key);
  }

  template <typename LookupKeyT>
  auto lookup(const LookupKeyT& lookup_key) const -> LookupKVResultT {
    return ViewT(*this).lookup(lookup_key);
  }

  template <typename LookupKeyT>
  auto operator[](const LookupKeyT& lookup_key) const -> ValueT* {
    return ViewT(*this)[lookup_key];
  }

  template <typename CallbackT>
  void forEach(CallbackT callback) {
    return ViewT(*this).forEach(callback);
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const {
    return {size(), small_size(), storage()};
  }

  template <typename LookupKeyT>
  auto insert(
      const LookupKeyT& lookup_key,
      typename std::__type_identity<llvm::function_ref<std::pair<
          KeyT*, ValueT*>(const LookupKeyT& lookup_key, void* key_storage,
                          void* value_storage)>>::type insert_cb)
      -> InsertKVResultT;

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
      typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
          const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>>::type
          insert_cb,
      llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
      -> InsertKVResultT;

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
  auto erase(const LookupKeyT& lookup_key) -> bool;

  void clear();

 protected:
  MapBase(int small_size, MapInternal::Storage* small_storage) {
    Init(small_size, small_storage);
  }
  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit MapBase(ssize_t alloc_size) {
    InitAlloc(alloc_size);
  }

  ~MapBase();

  auto size() const -> ssize_t { return impl_view_.size(); }
  auto storage() const -> MapInternal::Storage* { return impl_view_.storage_; }
  auto small_size() const -> int { return impl_view_.small_size(); }

  auto is_small() const -> bool { return small_size() >= 0; }

  void set_size(ssize_t size) {
    impl_view_.set_size(size);
  }
  void set_small_size(int small_size) {
    impl_view_.set_small_size(small_size);
  }

  auto storage() -> MapInternal::Storage*& { return impl_view_.storage_; }

  auto linear_keys() -> KeyT* { return MapInternal::getLinearKeys<KeyT>(storage()); }
  auto linear_values() -> ValueT* {
    return MapInternal::getLinearValues<KeyT, ValueT>(storage(), small_size());
  }

  auto groups_ptr() -> uint8_t* { return MapInternal::getGroupsPtr(storage()); }
  auto keys_ptr() -> KeyT* { return MapInternal::getKeys<KeyT>(storage(), size()); }
  auto values_ptr() -> ValueT* { return MapInternal::getValueFromKey<KeyT, ValueT>(keys_ptr(), size()); }

  void Init(int small_size, MapInternal::Storage* small_storage);
  void InitAlloc(ssize_t alloc_size);

  template <typename LookupKeyT>
  auto InsertIndexHashed(const LookupKeyT& lookup_key)
      -> std::pair<bool, ssize_t>;
  template <typename LookupKeyT>
  auto InsertIntoEmptyIndex(const LookupKeyT& lookup_key) -> ssize_t;
  template <typename LookupKeyT>
  auto InsertHashed(
      const LookupKeyT& lookup_key,
      llvm::function_ref<std::pair<KeyT*, ValueT*>(
          const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
          insert_cb) -> InsertKVResultT;
  template <typename LookupKeyT>
  auto InsertSmallLinear(
      const LookupKeyT& lookup_key,
      llvm::function_ref<std::pair<KeyT*, ValueT*>(
          const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
          insert_cb) -> InsertKVResultT;

  auto GrowAndRehash() -> uint8_t*;

  template <typename LookupKeyT>
  auto EraseSmallLinear(const LookupKeyT& lookup_key) -> bool;
  template <typename LookupKeyT>
  auto EraseHashed(const LookupKeyT& lookup_key) -> bool;

  ViewT impl_view_;
  int growth_budget_;
};

template <typename InputKeyT, typename InputValueT,
          ssize_t MinSmallSize =
              MapInternal::DefaultMinSmallSize<InputKeyT, InputValueT>()>
class alignas(MapInternal::ComputeMapAlignment<InputKeyT, InputValueT,
                                               MinSmallSize>()) Map
    : public MapBase<InputKeyT, InputValueT> {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using BaseT = MapBase<KeyT, ValueT>;
  using LookupKVResultT = MapInternal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = MapInternal::InsertKVResult<KeyT, ValueT>;

  Map() : BaseT(SmallSize, small_storage()) {}
  Map(const Map& arg) : Map() {
    arg.forEach([this](KeyT& k, ValueT& v) { insert(k, v); });
  }
  template <ssize_t OtherMinSmallSize>
  explicit Map(const Map<KeyT, ValueT, OtherMinSmallSize>& arg) : Map() {
    arg.forEach([this](KeyT& k, ValueT& v) { insert(k, v); });
  }
  Map(Map&& arg) = delete;

  void reset();

 private:
  static constexpr ssize_t SmallSize =
      MapInternal::ComputeSmallSize<KeyT, ValueT, MinSmallSize>();
  static constexpr bool UseLinearLookup =
      MapInternal::ShouldUseLinearLookup<KeyT>(SmallSize);

  static_assert(SmallSize >= 0, "Cannot have a negative small size!");

  using SmallSizeStorageT = MapInternal::SmallSizeStorage<KeyT, ValueT, UseLinearLookup, SmallSize>;

  // Validate a collection of invariants between the small size storage layout
  // and the dynamically computed storage layout. We need to do this after both
  // are complete but in the context of a specific key type, value type, and
  // small size, so here is the best place.
  static_assert(SmallSize == 0 || UseLinearLookup ||
                    (alignof(SmallSizeStorageT) ==
                     MapInternal::StorageAlignment<KeyT, ValueT>),
                "Small size buffer must have the same alignment as a heap "
                "allocated buffer.");
  static_assert(
      SmallSize == 0 ||
          (offsetof(SmallSizeStorageT, keys) ==
           (UseLinearLookup
                ? 0
                : MapInternal::ComputeKeyStorageOffset<KeyT>(SmallSize))),
      "Offset to keys in small size storage doesn't match computed offset!");
  static_assert(
      SmallSize == 0 ||
          (offsetof(SmallSizeStorageT, values) ==
           (UseLinearLookup
                ? 0
                : MapInternal::ComputeKeyStorageOffset<KeyT>(SmallSize)) +
               MapInternal::ComputeValueStorageOffset<KeyT, ValueT>(SmallSize)),
      "Offset from keys to values in small size storage doesn't match computed "
      "offset!");
  static_assert(SmallSize == 0 || UseLinearLookup ||
                    (sizeof(SmallSizeStorageT) ==
                     MapInternal::ComputeStorageSize<KeyT, ValueT>(SmallSize)),
                "The small size storage needs to match the dynamically "
                "computed storage size.");

  mutable MapInternal::SmallSizeStorage<KeyT, ValueT, UseLinearLookup,
                                        SmallSize>
      small_storage_;

  auto small_storage() const -> MapInternal::Storage* {
    return &small_storage_;
  }
};

// Implementation of the routines in `map_internal` that are used above.
namespace MapInternal {

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

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookupSmallLinear(const LookupKeyT& lookup_key, ssize_t size, int small_size,
                       Storage* storage) -> LookupKVResult<KeyT, ValueT> {
  KeyT* keys = getLinearKeys<KeyT>(storage);
  for (ssize_t i = 0; i < size; ++i) {
    KeyT *key = &keys[i];
    if (*key == lookup_key) {
      return {key, getValueFromKey<KeyT, ValueT>(
                       key, static_cast<unsigned>(small_size))};
    }
  }

  return {nullptr, nullptr};
}

inline auto computeProbeMaskFromSize(ssize_t size) -> size_t {
  assert(llvm::isPowerOf2_64(size) &&
         "Size must be a power of two for a hashed buffer!");
  // The probe mask needs to mask down to keep the index within
  // `groups_size`. Since `groups_size` is a power of two, this is equivalent to
  // `groups_size - 1`.
  return (size - 1) & ~GroupMask;
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
///   p(x,s) = (x + (s + s^2)/2) mod (Size / GroupSize)
///
/// This particular quadratic sequence will visit every value modulo the
/// provided size divided by the group size.
///
/// However, we compute it scaled to the group size constant G and have it visit
/// each G multiple modulo the size using the scaled formula:
/// 
///   p(x,s) = (x + (s + (s * s * G)/(G * G))/2) mod Size
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
    this->Start = start & Mask;
    this->Size = size;
#endif
  }

  void step() {
    Step += GroupSize;
    i = (i + Step) & Mask;
    assert(i == ((Start + ((Step + (Step * Step * GroupSize) /
                                       (GroupSize * GroupSize)) /
                           2)) %
                 Size) &&
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

inline auto computeHashIndex(size_t hash, uint8_t* groups) -> ssize_t {
  return hash ^ reinterpret_cast<uintptr_t>(groups);
}

template <typename LookupKeyT, typename KeyT>
inline auto lookupIndexHashed(LookupKeyT lookup_key, ssize_t size,
                              uint8_t* groups, KeyT* keys)
    -> ssize_t {
  //__asm volatile ("# LLVM-MCA-BEGIN hit");
  //__asm volatile ("# LLVM-MCA-BEGIN miss");
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = computeControlByte(hash);
  ssize_t hash_index = computeHashIndex(hash, groups);

  ProbeSequence s(hash_index, size);
  do {
    ssize_t group_index = s.getIndex();
    Group g = Group::load(groups, group_index);
    auto control_byte_matched_range = g.match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = group_index + *byte_it;
        if (LLVM_LIKELY(keys[index] == lookup_key)) {
          __builtin_assume(index > 0);
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

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto containsHashed(const LookupKeyT& lookup_key, ssize_t size,
                    Storage* storage) -> bool {
  //llvm::MutableArrayRef<uint8_t> groups = getGroups(storage, size);
  uint8_t* groups = getGroupsPtr(storage);
  KeyT* keys = getKeys<KeyT>(storage, size);
  return lookupIndexHashed(lookup_key, size, groups, keys) >= 0;
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto contains(const LookupKeyT& lookup_key, ssize_t size,
              int small_size, Storage* storage) -> bool {
  _mm_prefetch(storage, _MM_HINT_T2);
  if (ShouldUseLinearLookup<KeyT>(small_size)) {
    return containsSmallLinear<KeyT>(lookup_key, size, storage);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return containsHashed<KeyT, ValueT>(lookup_key, size, storage);
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookupHashed(LookupKeyT lookup_key, ssize_t size, Storage* storage)
    -> LookupKVResult<KeyT, ValueT> {
  uint8_t* groups = getGroupsPtr(storage);
  KeyT* keys = getKeys<KeyT>(storage, size);

  ssize_t index = lookupIndexHashed(lookup_key, size, groups, keys);
  if (index < 0) {
    return {nullptr, nullptr};
  }

  ValueT* values = getValueFromKey<KeyT, ValueT>(keys, size);
  return {&keys[index], &values[index]};
}

template <typename KeyT, typename ValueT, typename LookupKeyT>
auto lookup(const LookupKeyT& lookup_key, ssize_t size, int small_size,
            Storage* storage) -> LookupKVResult<KeyT, ValueT> {
  _mm_prefetch(storage, _MM_HINT_T2);
  if (ShouldUseLinearLookup<KeyT>(small_size)) {
    return lookupSmallLinear<KeyT, ValueT>(lookup_key, size, small_size, storage);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return lookupHashed<KeyT, ValueT>(lookup_key, size, storage);
}

} // namespace MapInternal

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::contains(const LookupKeyT& lookup_key) const -> bool {
  return MapInternal::contains<KeyT, ValueT>(lookup_key, size(),
                                             small_size(), storage_);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::lookup(const LookupKeyT& lookup_key) const
    -> LookupKVResultT {
  return MapInternal::lookup<KeyT, ValueT>(lookup_key, size(),
                                           small_size(), storage_);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::operator[](const LookupKeyT& lookup_key) const
    -> ValueT* {
  auto result = lookup(lookup_key);
  return result ? &result.getValue() : nullptr;
}

namespace MapInternal {

template <typename KeyT, typename ValueT, typename CallbackT>
void forEachLinear(ssize_t size, int small_size, Storage* storage,
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
  llvm::MutableArrayRef<uint8_t> groups = getGroups(storage, size);
  KeyT* keys = getKeys<KeyT>(storage, size);
  ValueT* values = getValues<KeyT, ValueT>(storage, size);

  for (ssize_t group_index = 0; group_index < size; group_index += GroupSize) {
    Group g = Group::load(groups, group_index);
    auto present_matched_range = g.matchPresent();
    if (!present_matched_range) {
      continue;
    }
    for (ssize_t byte_index : present_matched_range) {
      ssize_t index = group_index + byte_index;
      kv_callback(keys[index], values[index]);
    }

    group_callback(groups, group_index);
  }
}

template <typename KeyT, typename ValueT, typename CallbackT>
void forEach(ssize_t size, int small_size, Storage* storage,
             CallbackT callback) {
  _mm_prefetch(storage, _MM_HINT_T2);
  if (ShouldUseLinearLookup<KeyT>(small_size)) {
    forEachLinear<KeyT, ValueT>(size, small_size, storage, callback);
    return;
  }

  // Otherwise walk the non-empty slots in each control group.
  forEachHashed<KeyT, ValueT>(size, storage, callback, [](auto...) {});
}
}  // namespace MapInternal

template <typename KT, typename VT>
template <typename CallbackT>
void MapView<KT, VT>::forEach(CallbackT callback) {
    MapInternal::forEach<KeyT, ValueT>(size(), small_size(), storage_,
                                       callback);
  }


// Tries to insert the given lookup key into the map. Returns three pieces of
// data compressed into two registers (in order to avoid an in-memory return).
// These are the group pointer, a bool representing whether insertion is in fact
// required, and the byte index of either the found entry in the group or the
// slot of the group to insert into. The group pointer will be null if insertion
// isn't possible without growing.
template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::InsertIndexHashed(const LookupKeyT& lookup_key)
    -> std::pair<bool, ssize_t> {
  uint8_t* groups = groups_ptr();

  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = MapInternal::computeControlByte(hash);
  ssize_t hash_index = MapInternal::computeHashIndex(hash, groups);

  ssize_t group_with_deleted_index = -1;
  MapInternal::Group::MatchedByteRange<> deleted_matched_range;

  auto return_insert_at_index =
      [&](ssize_t index) -> std::pair<bool, ssize_t> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    groups[index] = control_byte;
    return {/*needs_insertion=*/true, index};
  };

  for (MapInternal::ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = MapInternal::Group::load(groups, group_index);

    auto control_byte_matched_range = g.match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = group_index + *byte_it;
        if (LLVM_LIKELY(keys_ptr()[index] == lookup_key)) {
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
    if (growth_budget_ == 0) {
      // Without room to grow, return that no group is viable but also set the
      // index to be negative. This ensures that a positive index is always
      // sufficient to determine that an existing was found.
      return {/*needs_insertion=*/true, -1};
    }

    return return_insert_at_index(group_index + *empty_matched_range.begin());
  }

  return return_insert_at_index(group_with_deleted_index +
                                *deleted_matched_range.begin());
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::InsertIntoEmptyIndex(const LookupKeyT& lookup_key) -> ssize_t {
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = MapInternal::computeControlByte(hash);
  uint8_t* groups = groups_ptr();
  ssize_t hash_index = MapInternal::computeHashIndex(hash, groups);

  for (MapInternal::ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = MapInternal::Group::load(groups, group_index);

    if (auto empty_matched_range = g.matchEmpty()) {
      ssize_t index = group_index + *empty_matched_range.begin();
      groups[index] = control_byte;
      return index;
    }

    // Otherwise we continue probing.
  }
}

namespace MapInternal {

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
}  // namespace MapInternal

template <typename KeyT, typename ValueT>
auto MapBase<KeyT, ValueT>::GrowAndRehash() -> uint8_t* {
  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  MapBase<KeyT, ValueT> new_map(MapInternal::computeNewSize(size()));
  KeyT* new_keys = new_map.keys_ptr();
  ValueT* new_values = new_map.values_ptr();

  forEach([&](KeyT& old_key, ValueT& old_value) {
    ssize_t index = new_map.InsertIntoEmptyIndex(old_key);
    --new_map.growth_budget_;
    new (&new_keys[index]) KeyT(std::move(old_key));
    old_key.~KeyT();
    new (&new_values[index]) ValueT(std::move(old_value));
    old_value.~ValueT();
  });
  assert(new_map.growth_budget_ >= 0 &&
         "Must still have a growth budget after rehash!");

  if (!is_small()) {
    // Old isn't a small buffer, so we need to deallocate it.
    MapInternal::deallocateStorage<KeyT, ValueT>(storage(), size());
  }

  // Now that we've fully built the new, grown structures, replace the entries
  // in the data structure. At this point we can be certain to not clobber
  // anything aliasing a small buffer.
  impl_view_ = new_map.impl_view_;
  growth_budget_ = new_map.growth_budget_;

  // Set the small size to -1 as we're not small any more. Nothing short of the
  // original map (which has the small size as a constant) can reset to the
  // small size so we don't need to preserve what this was once large.
  set_small_size(-1);

  // Prevent the interim new map object from doing anything when destroyed as
  // we've taken over it's internals.
  new_map.storage() = nullptr;
  new_map.set_size(0);

  // We return the newly allocated groups for immediate use by the caller.
  return groups_ptr();
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::InsertHashed(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb) -> InsertKVResultT {
  ssize_t index = -1;
  // Try inserting if we have storage at all.
  if (size() > 0) {
    bool needs_insertion;
    std::tie(needs_insertion, index) = InsertIndexHashed(lookup_key);
    if (!needs_insertion) {
      assert(index >= 0 &&
             "Must have a valid group when we find an existing entry.");
      return {false, keys_ptr()[index], values_ptr()[index]};
    }
  }

  if (index < 0) {
    assert(
        growth_budget_ == 0 &&
        "Shouldn't need to grow the table until we exhaust our growth budget!");

    GrowAndRehash();
    // Directly insert into an empty index as we know we have one.
    index = InsertIntoEmptyIndex(lookup_key);
  }

  assert(index >= 0 && "Should have a group to insert into now.");
  assert(growth_budget_ >= 0 && "Cannot insert with zero budget!");
  --growth_budget_;

  KeyT* k;
  ValueT* v;
  std::tie(k, v) =
      insert_cb(lookup_key, &keys_ptr()[index], &values_ptr()[index]);
  return {true, *k, *v};
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::InsertSmallLinear(
    const LookupKeyT& lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage, void* value_storage)>
        insert_cb)
    -> InsertKVResultT {
  assert(is_small() && "Must be using the small size.");

  KeyT* keys = linear_keys();
  ValueT* values = linear_values();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    if (keys[i] == lookup_key) {
      return {false, keys[i], values[i]};
    }
  }

  size_t index = size();

  // We need to insert. First see if we have space.
  if (size() < static_cast<unsigned>(small_size())) {
    // We can do the easy linear insert, just increment the size.
    set_size(size() + 1);
  } else {
    // No space for a linear insert so grow into a hash table and then do
    // a hashed insert.
    GrowAndRehash();
    index = InsertIntoEmptyIndex(lookup_key);
    --growth_budget_;
    keys = keys_ptr();
    values = values_ptr();
  }
  KeyT* k;
  ValueT* v;
  std::tie(k, v) = insert_cb(lookup_key, &keys[index], &values[index]);
  return {true, *k, *v};
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::insert(
    const LookupKeyT& lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage,
        void* value_storage)>>::type insert_cb) -> InsertKVResultT {
  _mm_prefetch(storage(), _MM_HINT_T2);
  if (MapInternal::ShouldUseLinearLookup<KeyT>(small_size())) {
    return InsertSmallLinear(lookup_key, insert_cb);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return InsertHashed(lookup_key, insert_cb);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::update(
    const LookupKeyT& lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        const LookupKeyT& lookup_key, void* key_storage,
        void* value_storage)>>::type insert_cb,
    llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
    -> InsertKVResultT {
  auto i_result =
      insert(lookup_key, insert_cb);

  if (i_result.isInserted()) {
    return i_result;
  }

  ValueT& v = update_cb(i_result.getKey(), i_result.getValue());
  return {false, i_result.getKey(), v};
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::EraseSmallLinear(const LookupKeyT& lookup_key)
    -> bool {
  KeyT* keys = linear_keys();
  ValueT* values = linear_values();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    if (keys[i] == lookup_key) {
      // Found the key, so clobber this entry with the last one and decrease
      // the size by one.
      set_size(size() - 1);
      keys[i] = std::move(keys[size()]);
      keys[size()].~KeyT();
      values[i] = std::move(values[size()]);
      values[size()].~ValueT();
      return true;
    }
  }

  return false;
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::EraseHashed(const LookupKeyT& lookup_key) -> bool {
  uint8_t* groups = groups_ptr();
  KeyT* keys = keys_ptr();

  ssize_t index = MapInternal::lookupIndexHashed(lookup_key, size(), groups, keys);
  if (index < 0) {
    return false;
  }

  KeyT* key = &keys[index];
  key->~KeyT();
  ValueT* value = values_ptr();
  value->~ValueT();

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  ssize_t group_index = index & ~MapInternal::GroupMask;
  auto g = MapInternal::Group::load(groups, group_index);
  auto empty_matched_range = g.matchEmpty();
  if (empty_matched_range) {
    groups[index] = MapInternal::Group::Empty;
    ++growth_budget_;
  } else {
    groups[index] = MapInternal::Group::Deleted;
  }
  return true;
}

template <typename KeyT, typename ValueT> template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::erase(const LookupKeyT& lookup_key) -> bool {
  if (MapInternal::ShouldUseLinearLookup<KeyT>(small_size())) {
    return EraseSmallLinear(lookup_key);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return EraseHashed(lookup_key);
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::clear() {
  auto destroy_cb = [](KeyT& k, ValueT& v) {
    // Destroy this key and value.
    k.~KeyT();
    v.~ValueT();
  };

  if (MapInternal::ShouldUseLinearLookup<KeyT>(small_size())) {
    // Destroy all the keys and values.
    MapInternal::forEachLinear<KeyT, ValueT>(size(), small_size(), storage(), destroy_cb);

    // Now reset the size to zero and we'll start again inserting into the
    // beginning of the small linear buffer.
    set_size(0);
    return;
  }

  // Otherwise walk the non-empty slots in the control group destroying each
  // one and clearing out the group.
  MapInternal::forEachHashed<KeyT, ValueT>(
      size(), storage(), destroy_cb,
      [](llvm::MutableArrayRef<uint8_t> groups, ssize_t group_index) {
        // Clear the group.
        std::memset(groups.data() + group_index, 0, MapInternal::GroupSize);
      });

  // And reset the growth budget.
  growth_budget_ = MapInternal::growthThresholdForSize(size());
}

template <typename KeyT, typename ValueT>
MapBase<KeyT, ValueT>::~MapBase() {
  // Nothing to do when in the un-allocated and unused state.
  if (size() == 0) {
    return;
  }
  
  // Destroy all the keys and values.
  forEach([](KeyT& k, ValueT& v) {
    k.~KeyT();
    v.~ValueT();
  });

  // If small, nothing to deallocate.
  if (is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the object.
  MapInternal::deallocateStorage<KeyT, ValueT>(storage(), size());
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::Init(int small_size_arg, MapInternal::Storage* small_storage) {
  set_size(small_size_arg);
  set_small_size(small_size_arg);

  storage() = small_storage;

  if (MapInternal::ShouldUseLinearLookup<KeyT>(small_size_arg)) {
    // We use size to mean empty when doing linear lookups.
    set_size(0);
    // Growth budget isn't relevant as long as we're doing linear lookups.
    growth_budget_ = 0;
    return;
  }

  // We're not using linear lookups in the small size, so initialize it as
  // an initial hash table.
  growth_budget_ = MapInternal::growthThresholdForSize(small_size_arg);
  llvm::MutableArrayRef<uint8_t> groups = getGroups(small_storage, small_size_arg);
  std::memset(groups.data(), 0, groups.size());
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::InitAlloc(ssize_t alloc_size) {
  set_size(alloc_size);
  set_small_size(-1);
  storage() = MapInternal::allocateStorage<KeyT, ValueT>(size());
  std::memset(groups_ptr(), 0, size());
  growth_budget_ = MapInternal::growthThresholdForSize(size());
}

template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
void Map<KeyT, ValueT, MinSmallSize>::reset() {
  // Nothing to do when in the un-allocated and unused state.
  if (this->size() == 0) {
    return;
  }

  // If in the small rep, just clear the objects. 
  if (this->is_small()) {
    this->clear();
    return;
  }

  // Otherwise do the first part of the clear to destroy all the elements.
  this->forEach([](KeyT& k, ValueT& v) {
    k.~KeyT();
    v.~ValueT();
  });

  // Deallocate the buffer.
  MapInternal::deallocateStorage<KeyT, ValueT>(this->storage(), this->size());

  // Re-initialize the whole thing.
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif
