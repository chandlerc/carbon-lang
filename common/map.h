// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_MAP_H_
#define CARBON_COMMON_MAP_H_

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

#include "common/check.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/AlignOf.h"
#include "llvm/Support/Alignment.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/ReverseIteration.h"
#include "llvm/Support/type_traits.h"

// Detect whether we can use SIMD accelerated implementations of the control
// groups.
#if defined(__SSSE3__)
#include <x86intrin.h>
#define CARBON_USE_X86_SIMD_CONTROL_GROUP 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define CARBON_USE_NEON_SIMD_CONTROL_GROUP 1
#endif

namespace Carbon {

template <typename KeyT, typename ValueT>
class MapView;
template <typename KeyT, typename ValueT>
class MapBase;
template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
class Map;

namespace MapInternal {

template <typename KeyT, typename ValueT>
class LookupKVResult {
 public:
  LookupKVResult() = default;
  LookupKVResult(KeyT* key, ValueT* value) : key_(key), value_(value) {}

  explicit operator bool() const { return key_ != nullptr; }

  auto key() const -> KeyT& { return *key_; }
  auto value() const -> ValueT& { return *value_; }

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

  auto is_inserted() const -> bool { return key_and_inserted_.getInt(); }

  auto key() const -> KeyT& { return *key_and_inserted_.getPointer(); }
  auto value() const -> ValueT& { return *value_; }

 private:
  llvm::PointerIntPair<KeyT*, 1, bool> key_and_inserted_;
  ValueT* value_ = nullptr;
};

template <typename MaskT, int Shift = 0, MaskT ZeroMask = 0>
class BitIndexRange {
 public:
  class Iterator
      : public llvm::iterator_facade_base<Iterator, std::forward_iterator_tag,
                                          ssize_t, ssize_t> {
   public:
    Iterator() = default;
    explicit Iterator(MaskT mask) : mask_(mask) {}

    auto operator==(const Iterator& rhs) const -> bool {
      return mask_ == rhs.mask_;
    }

    auto operator*() -> ssize_t& {
      CARBON_DCHECK(mask_ != 0) << "Cannot get an index from a zero mask!";
      __builtin_assume(mask_ != 0);
      index_ = static_cast<size_t>(llvm::countr_zero(mask_)) >> Shift;
      return index_;
    }

    auto operator++() -> Iterator& {
      CARBON_DCHECK(mask_ != 0) << "Must not increment past the end!";
      __builtin_assume(mask_ != 0);
      mask_ &= (mask_ - 1);
      return *this;
    }

   private:
    ssize_t index_;
    MaskT mask_ = 0;
  };

  BitIndexRange() = default;
  explicit BitIndexRange(MaskT mask) : mask_(mask) {}

  explicit operator bool() const { return !empty(); }
  auto empty() const -> bool {
    CARBON_DCHECK((mask_ & ZeroMask) == 0) << "Unexpected non-zero bits!";
    __builtin_assume((mask_ & ZeroMask) == 0);
    return mask_ == 0;
  }

  auto begin() const -> Iterator { return Iterator(mask_); }
  auto end() const -> Iterator { return Iterator(); }

 private:
  MaskT mask_ = 0;
};

#if CARBON_USE_X86_SIMD_CONTROL_GROUP
// An X86 SIMD optimized control group representation. This uses a 128-bit
// vector register to implement the control group. While this could also be
// expanded to 256-bit vector widths on sufficiently modern x86 processors, that
// doesn't provide an especially large performance benefit. Largely, it would
// allow increasing load factor. But a major goal is to keep the load factor and
// other benefits of the control group design while minimizing latency of
// various critical path operations, and larger control groups fundamentally
// increase the cache pressure for the critical path.
struct X86Group {
  // Each control byte can have special values. All special values have the
  // most significant bit set to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  using MatchedRange =
      BitIndexRange<uint32_t, /*Shift=*/0, /*ZeroMask=*/0xFFFF0000>;

  __m128i byte_vec = {};

  static auto Load(uint8_t* groups, ssize_t index) -> X86Group {
    X86Group g;
    g.byte_vec = _mm_load_si128(reinterpret_cast<__m128i*>(groups + index));
    return g;
  }

  auto Match(uint8_t match_byte) const -> MatchedRange {
    auto match_byte_vec = _mm_set1_epi8(match_byte);
    auto match_byte_cmp_vec = _mm_cmpeq_epi8(byte_vec, match_byte_vec);
    uint32_t mask = _mm_movemask_epi8(match_byte_cmp_vec);
    return MatchedRange(mask);
  }

  auto MatchEmpty() const -> MatchedRange { return Match(Empty); }

  auto MatchDeleted() const -> MatchedRange { return Match(Deleted); }

  auto MatchPresent() const -> MatchedRange {
    // We arrange the byte vector for present bytes so that we can directly
    // extract it as a mask.
    return MatchedRange(_mm_movemask_epi8(byte_vec));
  }
};
#endif

#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
// An ARM NEON optimized control group. This is the same size and in fact layout
// as the portable group, but largely uses NEON operations to implement the
// logic on an 8-byte vector.
struct NeonGroup {
  // Each control byte can have special values. All special values have the
  // most significant bit set to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  using MatchedRange = BitIndexRange<uint64_t, /*Shift=*/3>;

  uint8x8_t byte_vec = {};

  static auto Load(uint8_t* groups, ssize_t index) -> NeonGroup {
    NeonGroup g;
    g.byte_vec = vld1_u8(groups + index);
    return g;
  }

  auto Match(uint8_t match_byte) const -> MatchedRange {
    auto match_byte_vec = vdup_n_u8(match_byte);
    auto match_byte_cmp_vec = vceq_u8(byte_vec, match_byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchedRange(mask);
  }

  auto MatchEmpty() const -> MatchedRange {
    auto match_byte_cmp_vec = vceqz_u8(byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchedRange(mask);
  }

  auto MatchDeleted() const -> MatchedRange { return Match(Deleted); }

  auto MatchPresent() const -> MatchedRange {
    static constexpr uint64_t MSBs = 0x8080'8080'8080'8080ULL;
    uint64_t mask;
    std::memcpy(&mask, &byte_vec, sizeof(byte_vec));
    mask &= MSBs;
    return MatchedRange(mask);
  }
};
#endif

struct PortableGroup {
  // Each control byte can have special values. All special values have the
  // most significant bit set to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  // Constants used to implement the various bit manipulation tricks.
  static constexpr uint64_t LSBs = 0x0101'0101'0101'0101ULL;
  static constexpr uint64_t MSBs = 0x8080'8080'8080'8080ULL;

  using MatchedRange = BitIndexRange<uint64_t, 3>;

  uint64_t group = {};

  static auto Load(uint8_t* groups, ssize_t index) -> PortableGroup {
    PortableGroup g;
    std::memcpy(&g.group, groups + index, sizeof(group));
    return g;
  }

  auto Match(uint8_t match_byte) const -> MatchedRange {
    // This algorithm only works for matching *present* bytes. We leverage the
    // set high bit in the present case as part of the algorithm. The whole
    // algorithm has a critical path height of 4 operations, and does 6
    // operations total:
    //
    //          group | MSBs    LSBs * match_byte
    //                 \            /
    //                 mask ^ pattern
    //                      |
    // group & MSBs    MSBs - mask
    //        \            /
    //    group_MSBs & mask
    //
    // While it is superficially similar to the "find zero bytes in a word" bit
    // math trick, it is different because this is designed to
    // have no false positives and perfectly produce 0x80 for matching bytes and
    // 0x00 for non-matching bytes. This is do-able because we constrain to only
    // handle present matches which only require testing 7 bits and have a
    // particular layout.
    CARBON_DCHECK(match_byte & 0b1000'0000)
        << llvm::formatv("{0:b}", match_byte);
    // Set the high bit of every byte to `1`. The match byte always has this bit
    // set as well, which ensures the xor below, in addition to zeroing the byte
    // that matches, also clears the high bit of every byte.
    uint64_t mask = group | MSBs;
    // Broadcast the match byte to all bytes.
    uint64_t pattern = LSBs * match_byte;
    // Xor the broadcast pattern, making matched bytes become zero bytes.
    mask = mask ^ pattern;
    // Subtract the mask bytes from `0x80` bytes so that any non-zero mask byte
    // clears the high byte but zero leaves it intact.
    mask = MSBs - mask;
    // Mask down to the high bits, but only those in the original group.
    mask &= (group & MSBs);
#ifndef NDEBUG
    const auto* group_bytes = reinterpret_cast<const uint8_t*>(&group);
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, sizeof(group))) {
      uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
      if (group_bytes[byte_index] == match_byte) {
        CARBON_DCHECK(byte == 0x80)
            << "Should have a high bit set for a matched byte, found: "
            << llvm::formatv("{0:x}", byte);
      } else {
        CARBON_DCHECK(byte == 0)
            << "Should have no bits set for an unmatched byte, found: "
            << llvm::formatv("{0:x}", byte);
      }
    }
#endif
    return MatchedRange(mask);
  }

  auto MatchEmpty() const -> MatchedRange {
    // Materialize the group into a word.
    uint64_t mask = group | (group << 7);
    mask = ~mask & MSBs;
#ifndef NDEBUG
    const auto* group_bytes = reinterpret_cast<const uint8_t*>(&group);
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, sizeof(group))) {
      uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
      if (group_bytes[byte_index] == Empty) {
        CARBON_DCHECK(byte == 0x80)
            << "Should have a high bit set for a matched byte, found: "
            << llvm::formatv("{0:x}", byte);
      } else {
        CARBON_DCHECK(byte == 0)
            << "Should have no bits set for an unmatched byte, found: "
            << llvm::formatv("{0:x}", byte);
      }
    }
#endif
    return MatchedRange(mask);
  }

  auto MatchDeleted() const -> MatchedRange {
    // Materialize the group into a word.
    uint64_t mask = group | (~group << 7);
    mask = ~mask & MSBs;
#ifndef NDEBUG
    const auto* group_bytes = reinterpret_cast<const uint8_t*>(&group);
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, sizeof(group))) {
      uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
      if (group_bytes[byte_index] == Deleted) {
        CARBON_DCHECK(byte == 0x80)
            << "Should have a high bit set for a matched byte, found: "
            << llvm::formatv("{0:x}", byte);
      } else {
        CARBON_DCHECK(byte == 0)
            << "Should have no bits set for an unmatched byte, found: "
            << llvm::formatv("{0:x}", byte);
      }
    }
#endif
    return MatchedRange(mask);
  }

  auto MatchPresent() const -> MatchedRange {
    // Materialize the group into a word.
    uint64_t mask = group & MSBs;
#ifndef NDEBUG
    const auto* group_bytes = reinterpret_cast<const uint8_t*>(&group);
    for (ssize_t byte_index : llvm::seq<ssize_t>(0, sizeof(group))) {
      uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
      if (group_bytes[byte_index] & 0b1000'0000U) {
        CARBON_DCHECK(byte == 0x80)
            << "Should have a high bit set for a present byte, found: "
            << llvm::formatv("{0:x}", byte);
      } else {
        CARBON_DCHECK(byte == 0)
            << "Should have no bits set for a not-present byte, found: "
            << llvm::formatv("{0:x}", byte);
      }
    }
#endif
    return MatchedRange(mask);
  }
};

#if CARBON_USE_X86_SIMD_CONTROL_GROUP
using Group = X86Group;
#elif CARBON_USE_NEON_SIMD_CONTROL_GROUP
using Group = NeonGroup;
#else
using Group = PortableGroup;
#endif

constexpr ssize_t GroupSize = sizeof(Group);
static_assert(llvm::isPowerOf2_64(GroupSize),
              "The group size must be a constant power of two so dividing by "
              "it is a simple shift.");
constexpr ssize_t GroupMask = GroupSize - 1;

[[clang::always_inline]] inline void Prefetch(const void* address) {
  // Currently we just hard code a single "low" temporal locality prefetch as
  // we're primarily expecting a brief use of the storage and then to return to
  // application code.
  __builtin_prefetch(address, /*read*/ 0, /*low-locality*/ 1);
}

// We use pointers to this empty class to model the pointer to a dynamically
// allocated structure of arrays with the groups, keys, and values.
//
// This also lets us define statically allocated storage as subclasses.
struct Storage {};

template <typename KeyT, typename ValueT>
constexpr ssize_t StorageAlignment = std::max<ssize_t>(
    {GroupSize, alignof(Group), alignof(KeyT), alignof(ValueT)});

template <typename KeyT>
constexpr auto ComputeKeyStorageOffset(ssize_t size) -> ssize_t {
  // There are `size` control bytes plus any alignment needed for the key type.
  return llvm::alignTo<alignof(KeyT)>(size);
}

template <typename KeyT, typename ValueT>
constexpr auto ComputeValueStorageOffset(ssize_t size) -> ssize_t {
  // Skip the keys themselves.
  ssize_t offset = sizeof(KeyT) * size;

  // If the value type alignment is smaller than the key's, we're done.
  if constexpr (alignof(ValueT) <= alignof(KeyT)) {
    return offset;
  }

  // Otherwise, skip the alignment for the value type.
  return llvm::alignTo<alignof(ValueT)>(offset);
}

template <typename KeyT, typename ValueT>
constexpr auto ComputeStorageSize(ssize_t size) -> ssize_t {
  return ComputeKeyStorageOffset<KeyT>(size) +
         ComputeValueStorageOffset<KeyT, ValueT>(size) + sizeof(ValueT) * size;
}

constexpr ssize_t CachelineSize = 64;

template <typename KeyT>
constexpr auto NumKeysInCacheline() -> int {
  return CachelineSize / sizeof(KeyT);
}

template <typename KeyT, typename ValueT>
constexpr auto DefaultMinSmallSize() -> ssize_t {
  return (CachelineSize - 3 * sizeof(void*)) / (sizeof(KeyT) + sizeof(ValueT));
}

template <typename KeyT>
constexpr auto ShouldUseLinearLookup(int small_size) -> bool {
  // return false;
  return small_size >= 0 && small_size <= NumKeysInCacheline<KeyT>();
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

template <typename KeyT, typename ValueT, bool UseLinearLookup,
          ssize_t SmallSize>
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
  // minimum here or an exact figure.
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

}  // namespace MapInternal

template <typename InputKeyT, typename InputValueT>
class MapView {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using LookupKVResultT = typename MapInternal::LookupKVResult<KeyT, ValueT>;

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResultT;

  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const -> ValueT*;

  template <typename CallbackT>
  void ForEach(CallbackT callback);

 private:
  template <typename MapKeyT, typename MapValueT, ssize_t MinSmallSize>
  friend class Map;
  friend class MapBase<KeyT, ValueT>;

  MapView() = default;
  MapView(ssize_t size, bool is_linear, ssize_t small_size,
          MapInternal::Storage* storage)
      : storage_(storage) {
    SetSize(size);
    if (is_linear) {
      SetLinearValueOffset(small_size);
    } else {
      MakeNonLinear();
    }
  }

  int64_t packed_size_;
  MapInternal::Storage* storage_;

  auto size() const -> ssize_t { return static_cast<uint32_t>(packed_size_); }

  auto is_linear() const -> bool { return packed_size_ >= 0; }
  auto linear_value_offset() const -> ssize_t {
    assert(is_linear() && "No linear offset when not linear!");
    return static_cast<uint32_t>(packed_size_ >> 32);
  }
  auto linear_keys() const -> KeyT* {
    assert(is_linear() && "No linear keys when not linear!");
    return reinterpret_cast<KeyT*>(storage_);
  }
  auto linear_values() const -> ValueT* {
    assert(is_linear() && "No linear values when not linear!");
    return reinterpret_cast<ValueT*>(
        reinterpret_cast<unsigned char*>(storage_) + linear_value_offset());
  }
  auto linear_value_from_key(KeyT* key) const -> ValueT* {
    assert(is_linear() && "No linear values when not linear!");
    return linear_values() + (key - linear_keys());
  }

  auto groups_ptr() const -> uint8_t* {
    assert(!is_linear() && "No groups when linear!");
    return reinterpret_cast<uint8_t*>(storage_);
  }
  auto keys_ptr() const -> KeyT* {
    assert(!is_linear() && "No grouped keys when linear!");
    assert(llvm::isPowerOf2_64(size()) &&
           "Size must be a power of two for a hashed buffer!");
    assert(size() == MapInternal::ComputeKeyStorageOffset<KeyT>(size()) &&
           "Cannot be more aligned than a power of two.");
    return reinterpret_cast<KeyT*>(reinterpret_cast<unsigned char*>(storage_) +
                                   size());
  }
  auto values_ptr() const -> ValueT* {
    assert(!is_linear() && "No grouped values when linear!");
    return reinterpret_cast<ValueT*>(
        reinterpret_cast<unsigned char*>(keys_ptr()) +
        MapInternal::ComputeValueStorageOffset<KeyT, ValueT>(size()));
  }

  template <typename LookupKeyT>
  inline auto ContainsHashed(LookupKeyT lookup_key) const -> bool;
  template <typename LookupKeyT>
  inline auto LookupSmallLinear(LookupKeyT lookup_key) const -> LookupKVResultT;
  template <typename LookupKeyT>
  inline auto LookupHashed(LookupKeyT lookup_key) const -> LookupKVResultT;

  template <typename CallbackT>
  void ForEachLinear(CallbackT callback);
  template <typename KVCallbackT, typename GroupCallbackT>
  void ForEachHashed(KVCallbackT kv_callback, GroupCallbackT group_callback);

  void SetSize(ssize_t size) {
    assert(size >= 0 && "Cannot have a negative size!");
    assert(size <= INT_MAX && "Only 32-bit sizes are supported!");
    packed_size_ &= -1ULL << 32;
    packed_size_ |= size & ((1LL << 32) - 1);
  }
  void MakeNonLinear() { packed_size_ |= -1ULL << 32; }
  void SetLinearValueOffset(ssize_t small_size) {
    packed_size_ &= (1ULL << 32) - 1;
    packed_size_ |=
        static_cast<int64_t>(
            MapInternal::ComputeValueStorageOffset<KeyT, ValueT>(small_size))
        << 32;
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
  auto Contains(LookupKeyT lookup_key) const -> bool {
    return ViewT(*this).Contains(lookup_key);
  }

  template <typename LookupKeyT>
  auto Lookup(LookupKeyT lookup_key) const -> LookupKVResultT {
    return ViewT(*this).Lookup(lookup_key);
  }

  template <typename LookupKeyT>
  auto operator[](LookupKeyT lookup_key) const -> ValueT* {
    return ViewT(*this)[lookup_key];
  }

  template <typename CallbackT>
  void ForEach(CallbackT callback) {
    return ViewT(*this).ForEach(callback);
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewT() const { return impl_view_; }

  template <typename LookupKeyT>
  auto Insert(
      LookupKeyT lookup_key,
      typename std::__type_identity<llvm::function_ref<
          std::pair<KeyT*, ValueT*>(LookupKeyT lookup_key, void* key_storage,
                                    void* value_storage)>>::type insert_cb)
      -> InsertKVResultT;

  template <typename LookupKeyT>
  auto Insert(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResultT {
    return Insert(lookup_key,
                  [&new_v](LookupKeyT lookup_key, void* key_storage,
                           void* value_storage) -> std::pair<KeyT*, ValueT*> {
                    KeyT* k = new (key_storage) KeyT(lookup_key);
                    auto* v = new (value_storage) ValueT(std::move(new_v));
                    return {k, v};
                  });
  }

  template <typename LookupKeyT, typename ValueCallbackT>
  auto Insert(LookupKeyT lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResultT>::type {
    return Insert(
        lookup_key,
        [&value_cb](LookupKeyT lookup_key, void* key_storage,
                    void* value_storage) -> std::pair<KeyT*, ValueT*> {
          KeyT* k = new (key_storage) KeyT(lookup_key);
          auto* v = new (value_storage) ValueT(value_cb());
          return {k, v};
        });
  }

  template <typename LookupKeyT>
  auto Update(
      LookupKeyT lookup_key,
      typename std::__type_identity<llvm::function_ref<
          std::pair<KeyT*, ValueT*>(LookupKeyT lookup_key, void* key_storage,
                                    void* value_storage)>>::type insert_cb,
      llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
      -> InsertKVResultT;

  template <typename LookupKeyT, typename ValueCallbackT>
  auto Update(LookupKeyT lookup_key, ValueCallbackT value_cb) ->
      typename std::enable_if<
          !std::is_same<ValueT, ValueCallbackT>::value &&
              std::is_same<ValueT,
                           decltype(std::declval<ValueCallbackT>()())>::value,

          InsertKVResultT>::type {
    return Update(
        lookup_key,
        [&value_cb](LookupKeyT lookup_key, void* key_storage,
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
  auto Update(LookupKeyT lookup_key, ValueT new_v) -> InsertKVResultT {
    return Update(
        lookup_key,
        [&new_v](LookupKeyT lookup_key, void* key_storage,
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
  auto Erase(LookupKeyT lookup_key) -> bool;

  void Clear();

 protected:
  MapBase(int small_size, MapInternal::Storage* small_storage) {
    Init(small_size, small_storage);
  }
  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit MapBase(ssize_t alloc_size) { InitAlloc(alloc_size); }

  ~MapBase();

  auto size() const -> ssize_t { return impl_view_.size(); }
  auto storage() const -> MapInternal::Storage* { return impl_view_.storage_; }
  auto small_size() const -> ssize_t {
    return static_cast<unsigned>(small_size_);
  }

  auto is_small() const -> bool { return size() <= small_size(); }

  auto storage() -> MapInternal::Storage*& { return impl_view_.storage_; }

  auto linear_keys() -> KeyT* { return impl_view_.linear_keys(); }
  auto linear_values() -> ValueT* { return impl_view_.linear_values(); }

  auto groups_ptr() -> uint8_t* { return impl_view_.groups_ptr(); }
  auto keys_ptr() -> KeyT* { return impl_view_.keys_ptr(); }
  auto values_ptr() -> ValueT* { return impl_view_.values_ptr(); }

  void Init(int small_size, MapInternal::Storage* small_storage);
  void InitAlloc(ssize_t alloc_size);

  template <typename LookupKeyT>
  auto InsertIndexHashed(LookupKeyT lookup_key) -> std::pair<bool, ssize_t>;
  template <typename LookupKeyT>
  auto InsertIntoEmptyIndex(LookupKeyT lookup_key) -> ssize_t;
  template <typename LookupKeyT>
  auto InsertHashed(
      LookupKeyT lookup_key,
      llvm::function_ref<std::pair<KeyT*, ValueT*>(
          LookupKeyT lookup_key, void* key_storage, void* value_storage)>
          insert_cb) -> InsertKVResultT;
  template <typename LookupKeyT>
  auto InsertSmallLinear(
      LookupKeyT lookup_key,
      llvm::function_ref<std::pair<KeyT*, ValueT*>(
          LookupKeyT lookup_key, void* key_storage, void* value_storage)>
          insert_cb) -> InsertKVResultT;

  auto GrowAndRehash() -> uint8_t*;

  template <typename LookupKeyT>
  auto EraseSmallLinear(LookupKeyT lookup_key) -> bool;
  template <typename LookupKeyT>
  auto EraseHashed(LookupKeyT lookup_key) -> bool;

  ViewT impl_view_;
  int growth_budget_;
  int small_size_;
};

template <typename InputKeyT, typename InputValueT,
          ssize_t MinSmallSize =
              MapInternal::DefaultMinSmallSize<InputKeyT, InputValueT>()>
class Map : public MapBase<InputKeyT, InputValueT> {
 public:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewT = MapView<KeyT, ValueT>;
  using BaseT = MapBase<KeyT, ValueT>;
  using LookupKVResultT = MapInternal::LookupKVResult<KeyT, ValueT>;
  using InsertKVResultT = MapInternal::InsertKVResult<KeyT, ValueT>;

  Map() : BaseT(SmallSize, small_storage()) {}
  Map(const Map& arg) : Map() {
    arg.ForEach([this](KeyT& k, ValueT& v) { insert(k, v); });
  }
  template <ssize_t OtherMinSmallSize>
  explicit Map(const Map<KeyT, ValueT, OtherMinSmallSize>& arg) : Map() {
    arg.ForEach([this](KeyT& k, ValueT& v) { insert(k, v); });
  }
  Map(Map&& arg) = delete;

  void Reset();

 private:
  static constexpr ssize_t SmallSize =
      MapInternal::ComputeSmallSize<KeyT, ValueT, MinSmallSize>();
  static constexpr bool UseLinearLookup =
      MapInternal::ShouldUseLinearLookup<KeyT>(SmallSize);

  static_assert(SmallSize >= 0, "Cannot have a negative small size!");

  using SmallSizeStorageT =
      MapInternal::SmallSizeStorage<KeyT, ValueT, UseLinearLookup, SmallSize>;

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
inline auto ContainsSmallLinear(const LookupKeyT& lookup_key, ssize_t size,
                                KeyT* keys) -> bool {
  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    if (keys[i] == lookup_key) {
      return true;
    }
  }

  return false;
}

inline auto ComputeProbeMaskFromSize(ssize_t size) -> size_t {
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
    Mask = ComputeProbeMaskFromSize(size);
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

inline auto ComputeControlByte(size_t hash) -> uint8_t {
  // Mask one over the high bit so that engaged control bytes are easily
  // identified.
  return (hash >> (sizeof(hash) * 8 - 7)) | 0b10000000;
}

inline auto ComputeHashIndex(size_t hash, const void* ptr) -> ssize_t {
  return hash ^ reinterpret_cast<uintptr_t>(ptr);
}

template <typename KeyT, typename LookupKeyT>
[[clang::noinline]] auto LookupIndexHashed(LookupKeyT lookup_key, ssize_t size,
                                           Storage* storage) -> ssize_t {
  uint8_t* groups = reinterpret_cast<uint8_t*>(storage);
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = ComputeControlByte(hash);
  ssize_t hash_index = ComputeHashIndex(hash, groups);

  KeyT* keys =
      reinterpret_cast<KeyT*>(reinterpret_cast<unsigned char*>(storage) + size);
  ProbeSequence s(hash_index, size);
  do {
    ssize_t group_index = s.getIndex();
    Group g = Group::Load(groups, group_index);
    auto control_byte_matched_range = g.Match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = group_index + *byte_it;
        if (LLVM_LIKELY(keys[index] == lookup_key)) {
          __builtin_assume(index >= 0);
          return index;
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots and we're done probing.
    auto empty_byte_matched_range = g.MatchEmpty();
    if (LLVM_LIKELY(empty_byte_matched_range)) {
      return -1;
    }

    s.step();
  } while (LLVM_UNLIKELY(true));
}

}  // namespace MapInternal

template <typename KT, typename VT>
template <typename LookupKeyT>
inline auto MapView<KT, VT>::ContainsHashed(LookupKeyT lookup_key) const
    -> bool {
  return MapInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage_) >=
         0;
}

template <typename KT, typename VT>
template <typename LookupKeyT>
inline auto MapView<KT, VT>::LookupSmallLinear(LookupKeyT lookup_key) const
    -> LookupKVResultT {
  KeyT* key = linear_keys();
  KeyT* key_end = &key[size()];
  ValueT* value = linear_values();
  do {
    if (*key == lookup_key) {
      return {key, value};
    }
    ++key;
    ++value;
  } while (key < key_end);

  return {nullptr, nullptr};
}

template <typename KT, typename VT>
template <typename LookupKeyT>
inline auto MapView<KT, VT>::LookupHashed(LookupKeyT lookup_key) const
    -> LookupKVResultT {
  ssize_t index =
      MapInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage_);
  if (index < 0) {
    return {nullptr, nullptr};
  }

  return {&keys_ptr()[index], &values_ptr()[index]};
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::Contains(LookupKeyT lookup_key) const -> bool {
  MapInternal::Prefetch(storage_);
  if (is_linear()) {
    return MapInternal::ContainsSmallLinear<KeyT>(lookup_key, size(),
                                                  linear_keys());
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return ContainsHashed(lookup_key);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::Lookup(LookupKeyT lookup_key) const -> LookupKVResultT {
  MapInternal::Prefetch(storage_);
  if (is_linear()) {
    return LookupSmallLinear(lookup_key);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return LookupHashed(lookup_key);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapView<KT, VT>::operator[](LookupKeyT lookup_key) const -> ValueT* {
  auto result = Lookup(lookup_key);
  return result ? &result.value() : nullptr;
}

template <typename KeyT, typename ValueT>
template <typename CallbackT>
void MapView<KeyT, ValueT>::ForEachLinear(CallbackT callback) {
  KeyT* keys = linear_keys();
  ValueT* values = linear_values();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    callback(keys[i], values[i]);
  }
}

template <typename KeyT, typename ValueT>
template <typename KVCallbackT, typename GroupCallbackT>
void MapView<KeyT, ValueT>::ForEachHashed(KVCallbackT kv_callback,
                                          GroupCallbackT group_callback) {
  uint8_t* groups = groups_ptr();
  KeyT* keys = keys_ptr();
  ValueT* values = values_ptr();

  ssize_t local_size = size();
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += MapInternal::GroupSize) {
    auto g = MapInternal::Group::Load(groups, group_index);
    auto present_matched_range = g.MatchPresent();
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

template <typename KT, typename VT>
template <typename CallbackT>
void MapView<KT, VT>::ForEach(CallbackT callback) {
  MapInternal::Prefetch(storage_);
  if (is_linear()) {
    ForEachLinear(callback);
    return;
  }

  // Otherwise walk the non-empty slots in each control group.
  ForEachHashed(callback, [](auto...) {});
}

// Tries to insert the given lookup key into the map. Returns three pieces of
// data compressed into two registers (in order to avoid an in-memory return).
// These are the group pointer, a bool representing whether insertion is in fact
// required, and the byte index of either the found entry in the group or the
// slot of the group to insert into. The group pointer will be null if insertion
// isn't possible without growing.
template <typename KT, typename VT>
template <typename LookupKeyT>
[[clang::noinline]]
auto MapBase<KT, VT>::InsertIndexHashed(LookupKeyT lookup_key)
    -> std::pair<bool, ssize_t> {
  uint8_t* groups = groups_ptr();

  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = MapInternal::ComputeControlByte(hash);
  ssize_t hash_index = MapInternal::ComputeHashIndex(hash, groups);

  ssize_t group_with_deleted_index = -1;
  MapInternal::Group::MatchedRange deleted_matched_range;

  auto return_insert_at_index = [&](ssize_t index) -> std::pair<bool, ssize_t> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    groups[index] = control_byte;
    return {/*needs_insertion=*/true, index};
  };

  for (MapInternal::ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = MapInternal::Group::Load(groups, group_index);

    auto control_byte_matched_range = g.Match(control_byte);
    if (control_byte_matched_range) {
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
      deleted_matched_range = g.MatchDeleted();
      if (deleted_matched_range) {
        group_with_deleted_index = group_index;
      }
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // no empty slots. In that case, we'll continue probing.
    auto empty_matched_range = g.MatchEmpty();
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
[[clang::noinline]]
auto MapBase<KT, VT>::InsertIntoEmptyIndex(LookupKeyT lookup_key) -> ssize_t {
  size_t hash = llvm::hash_value(lookup_key);
  uint8_t control_byte = MapInternal::ComputeControlByte(hash);
  uint8_t* groups = groups_ptr();
  ssize_t hash_index = MapInternal::ComputeHashIndex(hash, groups);

  for (MapInternal::ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = MapInternal::Group::Load(groups, group_index);

    if (auto empty_matched_range = g.MatchEmpty()) {
      ssize_t index = group_index + *empty_matched_range.begin();
      groups[index] = control_byte;
      return index;
    }

    // Otherwise we continue probing.
  }
}

namespace MapInternal {

template <typename KeyT, typename ValueT>
inline auto AllocateStorage(ssize_t size) -> Storage* {
  ssize_t allocated_size = ComputeStorageSize<KeyT, ValueT>(size);
  return reinterpret_cast<Storage*>(__builtin_operator_new(
      allocated_size, std::align_val_t(StorageAlignment<KeyT, ValueT>),
      std::nothrow_t()));
}

template <typename KeyT, typename ValueT>
inline void DeallocateStorage(Storage* storage, ssize_t size) {
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

inline auto ComputeNewSize(ssize_t old_size) -> ssize_t {
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

inline auto GrowthThresholdForSize(ssize_t size) -> ssize_t {
  // We use a 7/8ths load factor to trigger growth.
  return size - size / 8;
}

}  // namespace MapInternal

template <typename KeyT, typename ValueT>
[[clang::noinline]]
auto MapBase<KeyT, ValueT>::GrowAndRehash() -> uint8_t* {
  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  MapBase<KeyT, ValueT> new_map(MapInternal::ComputeNewSize(size()));
  KeyT* new_keys = new_map.keys_ptr();
  ValueT* new_values = new_map.values_ptr();

  ForEach([&](KeyT& old_key, ValueT& old_value) {
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
    MapInternal::DeallocateStorage<KeyT, ValueT>(storage(), size());
  }

  // Now that we've fully built the new, grown structures, replace the entries
  // in the data structure. At this point we can be certain to not clobber
  // anything aliasing a small buffer.
  impl_view_ = new_map.impl_view_;
  growth_budget_ = new_map.growth_budget_;

  // Mark the map as non-linear.
  impl_view_.MakeNonLinear();

  // Prevent the ephemeral new map object from doing anything when destroyed as
  // we've taken over it's internals.
  new_map.storage() = nullptr;
  new_map.impl_view_.SetSize(0);

  // We return the newly allocated groups for immediate use by the caller.
  return groups_ptr();
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::InsertHashed(
    LookupKeyT lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>
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
    LookupKeyT lookup_key,
    llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>
        insert_cb) -> InsertKVResultT {
  KeyT* keys = linear_keys();
  ValueT* values = linear_values();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    if (keys[i] == lookup_key) {
      return {false, keys[i], values[i]};
    }
  }

  ssize_t old_size = size();
  ssize_t index = old_size;

  // We need to insert. First see if we have space.
  if (old_size < small_size()) {
    // We can do the easy linear insert, just increment the size.
    impl_view_.SetSize(old_size + 1);
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
auto MapBase<KT, VT>::Insert(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>>::type
        insert_cb) -> InsertKVResultT {
  Prefetch(storage());
  if (impl_view_.is_linear()) {
    return InsertSmallLinear(lookup_key, insert_cb);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return InsertHashed(lookup_key, insert_cb);
}

template <typename KT, typename VT>
template <typename LookupKeyT>
auto MapBase<KT, VT>::Update(
    LookupKeyT lookup_key,
    typename std::__type_identity<llvm::function_ref<std::pair<KeyT*, ValueT*>(
        LookupKeyT lookup_key, void* key_storage, void* value_storage)>>::type
        insert_cb,
    llvm::function_ref<ValueT&(KeyT& key, ValueT& value)> update_cb)
    -> InsertKVResultT {
  auto i_result = Insert(lookup_key, insert_cb);

  if (i_result.is_inserted()) {
    return i_result;
  }

  ValueT& v = update_cb(i_result.key(), i_result.value());
  return {false, i_result.key(), v};
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::EraseSmallLinear(LookupKeyT lookup_key) -> bool {
  KeyT* keys = linear_keys();
  ValueT* values = linear_values();
  for (ssize_t i : llvm::seq<ssize_t>(0, size())) {
    if (keys[i] == lookup_key) {
      // Found the key, so clobber this entry with the last one and decrease
      // the size by one.
      impl_view_.SetSize(size() - 1);
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
auto MapBase<KeyT, ValueT>::EraseHashed(LookupKeyT lookup_key) -> bool {
  ssize_t index =
      MapInternal::LookupIndexHashed<KeyT>(lookup_key, size(), storage());
  if (index < 0) {
    return false;
  }

  KeyT* keys = keys_ptr();
  keys[index].~KeyT();
  values_ptr()[index].~ValueT();

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  uint8_t* groups = groups_ptr();
  ssize_t group_index = index & ~MapInternal::GroupMask;
  auto g = MapInternal::Group::Load(groups, group_index);
  auto empty_matched_range = g.MatchEmpty();
  if (empty_matched_range) {
    groups[index] = MapInternal::Group::Empty;
    ++growth_budget_;
  } else {
    groups[index] = MapInternal::Group::Deleted;
  }
  return true;
}

template <typename KeyT, typename ValueT>
template <typename LookupKeyT>
auto MapBase<KeyT, ValueT>::Erase(LookupKeyT lookup_key) -> bool {
  if (impl_view_.is_linear()) {
    return EraseSmallLinear(lookup_key);
  }

  // Otherwise we dispatch to the hashed routine which is the same for small
  // and large.
  return EraseHashed(lookup_key);
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::Clear() {
  auto destroy_cb = [](KeyT& k, ValueT& v) {
    // Destroy this key and value.
    k.~KeyT();
    v.~ValueT();
  };

  if (impl_view_.is_linear()) {
    // Destroy all the keys and values.
    impl_view_.ForEachLinear(destroy_cb);

    // Now reset the size to zero and we'll start again inserting into the
    // beginning of the small linear buffer.
    impl_view_.SetSize(0);
    return;
  }

  // Otherwise walk the non-empty slots in the control group destroying each
  // one and clearing out the group.
  impl_view_.ForEachHashed(
      destroy_cb, [](uint8_t* groups, ssize_t group_index) {
        // Clear the group.
        std::memset(groups + group_index, 0, MapInternal::GroupSize);
      });

  // And reset the growth budget.
  growth_budget_ = MapInternal::GrowthThresholdForSize(size());
}

template <typename KeyT, typename ValueT>
MapBase<KeyT, ValueT>::~MapBase() {
  // Nothing to do when in the un-allocated and unused state.
  if (size() == 0) {
    return;
  }

  // Destroy all the keys and values.
  ForEach([](KeyT& k, ValueT& v) {
    k.~KeyT();
    v.~ValueT();
  });

  // If small, nothing to deallocate.
  if (is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  MapInternal::DeallocateStorage<KeyT, ValueT>(storage(), size());
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::Init(int small_size_arg,
                                 MapInternal::Storage* small_storage) {
  storage() = small_storage;
  small_size_ = small_size_arg;

  if (MapInternal::ShouldUseLinearLookup<KeyT>(small_size_arg)) {
    // We use size to mean empty when doing linear lookups.
    impl_view_.SetSize(0);
    impl_view_.SetLinearValueOffset(small_size_arg);
  } else {
    // We're not using linear lookups in the small size, so initialize it as
    // an initial hash table.
    impl_view_.SetSize(small_size_arg);
    impl_view_.MakeNonLinear();
    growth_budget_ = MapInternal::GrowthThresholdForSize(small_size_arg);
    std::memset(groups_ptr(), 0, small_size_arg);
  }
}

template <typename KeyT, typename ValueT>
void MapBase<KeyT, ValueT>::InitAlloc(ssize_t alloc_size) {
  assert(alloc_size > 0 && "Can only allocate positive size tables!");
  impl_view_.SetSize(alloc_size);
  impl_view_.MakeNonLinear();
  storage() = MapInternal::AllocateStorage<KeyT, ValueT>(alloc_size);
  std::memset(groups_ptr(), 0, alloc_size);
  growth_budget_ = MapInternal::GrowthThresholdForSize(alloc_size);
  small_size_ = 0;
}

template <typename KeyT, typename ValueT, ssize_t MinSmallSize>
void Map<KeyT, ValueT, MinSmallSize>::Reset() {
  // Nothing to do when in the un-allocated and unused state.
  if (this->size() == 0) {
    return;
  }

  // If in the small rep, just clear the objects.
  if (this->is_small()) {
    this->Clear();
    return;
  }

  // Otherwise do the first part of the clear to destroy all the elements.
  this->ForEach([](KeyT& k, ValueT& v) {
    k.~KeyT();
    v.~ValueT();
  });

  // Deallocate the buffer.
  MapInternal::DeallocateStorage<KeyT, ValueT>(this->storage(), this->size());

  // Re-initialize the whole thing.
  this->Init(SmallSize, small_storage());
}

}  // namespace Carbon

#endif  // CARBON_COMMON_MAP_H_
