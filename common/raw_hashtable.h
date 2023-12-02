// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_H_
#define CARBON_COMMON_RAW_HASHTABLE_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/hashing.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/bit.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

// Detect whether we can use SIMD accelerated implementations of the control
// groups.
#if defined(__SSSE3__)
#include <x86intrin.h>
#define CARBON_USE_X86_SIMD_CONTROL_GROUP 1
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#define CARBON_USE_NEON_SIMD_CONTROL_GROUP 1
#endif

// A namespace collecting a set of low-level utilities for building hashtable
// data structures. These should only be used as implementation details of
// higher-level data structure APIs.
//
// For example, see `set.h` for a hashtable-based set data structure, and
// `map.h` for a hashtable-based map data structure.
//
// The utilities in this namespace fall into a few categories:
//
// - Primitives to manage "groups" of hashtable entries that have densely packed
//   control bytes we can scan rapidly as a group, often using SIMD facilities
//   to process the entire group at once.
//
// - Tools to manipulate and work with the storage of offsets needed to
//   represent both key and key-value hashtables using these groups to organize
//   their entries.
//
// - Abstractions around efficiently probing across the hashtable consisting of
//   these "groups" of entries, and scanning within them to implement
//   traditional open-hashing hashtable operations.
//
// - Base classes to provide as much of the implementation of the user-facing
//   APIs as possible in a common way. This includes the most performance
//   sensitive code paths for the implementation of the data structures.
namespace Carbon::RawHashtable {

// We define a constant max group size. The particular group size used in
// practice may vary, but we want to have some upper bound that can be reliably
// used when allocating memory to ensure we don't create fractional groups and
// memory allocation is done consistently across the architectures.
constexpr ssize_t MaxGroupSize = 16;

// A global variable whose address is used as a seed. This allows ASLR to
// introduce some variation in hashtable ordering.
extern volatile std::byte global_addr_seed;

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

  template <int N>
  auto Test() const -> bool {
    return mask_ & (static_cast<MaskT>(1) << N);
  }

  explicit operator MaskT() const { return mask_; }

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

  auto Store(uint8_t* groups, ssize_t index) const -> void {
    _mm_store_si128(reinterpret_cast<__m128i*>(groups + index), byte_vec);
  }

  template <int Index>
  auto Set(uint8_t byte) -> void {
    byte_vec = _mm_insert_epi8(byte_vec, byte, Index);
  }

  auto ClearDeleted() -> void {
    // We cat zero every byte that isn't present.
    byte_vec = _mm_blendv_epi8(_mm_setzero_si128(), byte_vec, byte_vec);
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

  static constexpr uint64_t MSBs = 0x8080'8080'8080'8080ULL;

  using MatchedRange = BitIndexRange<uint64_t, /*Shift=*/3>;

  uint8x8_t byte_vec = {};

  static auto Load(uint8_t* groups, ssize_t index) -> NeonGroup {
    NeonGroup g;
    g.byte_vec = vld1_u8(groups + index);
    return g;
  }

  auto Store(uint8_t* groups, ssize_t index) const -> void {
    vst1_u8(groups + index, byte_vec);
  }

  auto ClearDeleted() -> void {
    // Compare less than zero of each byte to identify the present elements.
    uint8x8_t present_mask =
        vclt_s8(vreinterpret_s8_u8(byte_vec), vdup_n_s8(0));
    // And mask every other lane to zero.
    byte_vec = vand_u8(byte_vec, present_mask);
  }

  template <int Index>
  auto Set(uint8_t byte) -> void {
    byte_vec = vset_lane_u8(byte, byte_vec, Index);
  }

  auto Match(uint8_t match_byte) const -> MatchedRange {
    auto match_byte_vec = vdup_n_u8(match_byte);
    auto match_byte_cmp_vec = vceq_u8(byte_vec, match_byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchedRange(mask & MSBs);
  }

  auto MatchEmpty() const -> MatchedRange {
    auto match_byte_cmp_vec = vceqz_u8(byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchedRange(mask & MSBs);
  }

  auto MatchDeleted() const -> MatchedRange { return Match(Deleted); }

  auto MatchPresent() const -> MatchedRange {
    // Just directly extract the bytes as the MSB already marks presence.
    uint64_t mask = vreinterpret_u64_u8(byte_vec)[0];
    return MatchedRange(mask & MSBs);
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

  auto Store(uint8_t* groups, ssize_t index) const -> void {
    std::memcpy(groups + index, &group, sizeof(group));
  }

  template <int Index>
  auto Set(uint8_t byte) -> void {
    uint64_t incoming = static_cast<uint64_t>(byte) << Index * 8;
    group &= ~(static_cast<uint64_t>(0xff) << Index * 8);
    group |= incoming;
  }

  auto ClearDeleted() -> void { group &= (~LSBs | group >> 7); }

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
static_assert(GroupSize <= MaxGroupSize);
static_assert(MaxGroupSize % GroupSize == 0);
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

template <typename... Ts>
constexpr ssize_t StorageAlignment =
    std::max<ssize_t>({GroupSize, alignof(Group), alignof(Ts)...});

// Utility function to compute the offset from the storage pointer to the key
// array.
template <typename KeyT>
constexpr auto ComputeKeyStorageOffset(ssize_t size) -> ssize_t {
  // There are `size` control bytes plus any alignment needed for the key type.
  return llvm::alignTo<alignof(KeyT)>(size);
}

// Utility function to compute the offset from the storage pointer to the value
// array (assuming one exists). This is only valid to use in map-oriented
// derivations of the raw hashtables where we have both key and value storage.
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

// If allocating storage, allocate a minimum of one cacheline of group metadata
// and a minimum of one group.
constexpr ssize_t MinAllocatedSize = std::max<ssize_t>(64, MaxGroupSize);

template <typename KeyT>
constexpr static auto ComputeKeyStorageSize(ssize_t size) -> ssize_t {
  return ComputeKeyStorageOffset<KeyT>(size) + sizeof(KeyT) * size;
}

template <typename KeyT, typename ValueT>
constexpr static auto ComputeKeyValueStorageSize(ssize_t size) -> ssize_t {
  return ComputeKeyStorageOffset<KeyT>(size) +
         ComputeValueStorageOffset<KeyT, ValueT>(size) + sizeof(ValueT) * size;
}

// Template classes to form helper classes of small-size storage buffers with
// the correct layout. These can be used by either set-oriented hash tables or
// map-oriented to provide inline storage.
template <typename KeyT, ssize_t SmallSize>
struct SmallSizeKeyStorage;
template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct SmallSizeKeyValueStorage;

template <typename KeyT>
struct SmallSizeKeyStorage<KeyT, 0> : Storage {
  SmallSizeKeyStorage() {}
  union {
    KeyT keys[0];
  };
};
template <typename KeyT, typename ValueT>
struct SmallSizeKeyValueStorage<KeyT, ValueT, 0>
    : SmallSizeKeyStorage<KeyT, 0> {
  SmallSizeKeyValueStorage() {}
  union {
    ValueT values[0];
  };
};

template <typename KeyT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT>) SmallSizeKeyStorage : Storage {
  // Do early validation of the small size here.
  static_assert(llvm::isPowerOf2_64(SmallSize),
                "SmallSize must be a power of two for a hashed buffer!");
  static_assert(SmallSize >= MaxGroupSize,
                "We require all small sizes to multiples of the largest group "
                "size supported to ensure it can be used portably.  ");
  static_assert((SmallSize % MaxGroupSize) == 0,
                "Small size must be a multiple of the max group size supported "
                "so that we can allocate a whole number of groups.");
  // Implied by the max asserts above.
  static_assert(SmallSize >= GroupSize);
  static_assert((SmallSize % GroupSize) == 0);

  static constexpr ssize_t SmallNumGroups = SmallSize / GroupSize;
  static_assert(llvm::isPowerOf2_64(SmallNumGroups),
                "The number of groups must be a power of two when hashing!");

  SmallSizeKeyStorage() {
    // Validate a collection of invariants between the small size storage layout
    // and the dynamically computed storage layout. We need the key type to be
    // complete so we do this in the constructor body.
    static_assert(SmallSize == 0 ||
                      alignof(SmallSizeKeyStorage) == StorageAlignment<KeyT>,
                  "Small size buffer must have the same alignment as a heap "
                  "allocated buffer.");
    static_assert(
        SmallSize == 0 || (offsetof(SmallSizeKeyStorage, keys) ==
                           ComputeKeyStorageOffset<KeyT>(SmallSize)),
        "Offset to keys in small size storage doesn't match computed offset!");
    static_assert(SmallSize == 0 || sizeof(SmallSizeKeyStorage) ==
                                        ComputeKeyStorageSize<KeyT>(SmallSize),
                  "The small size storage needs to match the dynamically "
                  "computed storage size.");
  }

  Group groups[SmallNumGroups];

  union {
    KeyT keys[SmallSize];
  };
};
template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT, ValueT>) SmallSizeKeyValueStorage
    : RawHashtable::SmallSizeKeyStorage<KeyT, SmallSize> {
  SmallSizeKeyValueStorage() {
    static_assert(SmallSize == 0 ||
                      offsetof(SmallSizeKeyValueStorage, values) ==
                          (ComputeKeyStorageOffset<KeyT>(SmallSize) +
                           ComputeValueStorageOffset<KeyT, ValueT>(SmallSize)),
                  "Offset from keys to values in small size storage doesn't "
                  "match computed offset!");
    static_assert(SmallSize == 0 ||
                      sizeof(SmallSizeKeyValueStorage) ==
                          ComputeKeyValueStorageSize<KeyT, ValueT>(SmallSize),
                  "The small size storage needs to match the dynamically "
                  "computed storage size.");
  }

  union {
    ValueT values[SmallSize];
  };
};

// Base class to hold all the implementation details that can be completely
// isolated from the value type or even *if there is* a value type in the
// hashtable.
template <typename KeyT>
class RawHashtableKeyBase;

// Base class that encodes either the absence of a value or a value type.
template <typename KeyT, typename ValueT = void>
class RawHashtableBase;

template <typename InputKeyT>
class RawHashtableViewBase {
 protected:
  using KeyT = InputKeyT;

  friend class RawHashtableKeyBase<KeyT>;
  template <typename KeyT, typename ValueT>
  friend class RawHashtableBase;

  RawHashtableViewBase() = default;
  RawHashtableViewBase(ssize_t size, Storage* storage)
      : size_(size), storage_(storage) {}

  auto size() const -> ssize_t { return size_; }

  auto groups_ptr() const -> uint8_t* {
    return reinterpret_cast<uint8_t*>(storage_);
  }
  auto keys_ptr() const -> KeyT* {
    CARBON_DCHECK(size() == 0 || llvm::isPowerOf2_64(size()))
        << "Size must be a power of two for a hashed buffer!";
    CARBON_DCHECK(size() == ComputeKeyStorageOffset<KeyT>(size()))
        << "Cannot be more aligned than a power of two.";
    return reinterpret_cast<KeyT*>(reinterpret_cast<unsigned char*>(storage_) +
                                   size());
  }

  template <typename LookupKeyT>
  auto LookupIndexHashed(LookupKeyT lookup_key) const -> ssize_t;

  template <typename IndexCallbackT, typename GroupCallbackT>
  void ForEachIndex(IndexCallbackT index_callback,
                    GroupCallbackT group_callback);

  auto CountProbedKeys() const -> ssize_t;

  ssize_t size_;
  Storage* storage_;
};

template <typename InputKeyT>
class RawHashtableKeyBase {
 protected:
  using KeyT = InputKeyT;
  using ViewBaseT = RawHashtableViewBase<KeyT>;

  RawHashtableKeyBase(int small_size, Storage* small_storage) {
    Init(small_size, small_storage);
    small_size_ = small_size;
  }
  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit RawHashtableKeyBase(ssize_t arg_size, Storage* arg_storage) {
    Init(arg_size, arg_storage);
    small_size_ = 0;
  }

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewBaseT() const { return impl_view_; }

  auto size() const -> ssize_t { return impl_view_.size_; }
  auto size() -> ssize_t& { return impl_view_.size_; }
  auto storage() const -> Storage* { return impl_view_.storage_; }
  auto storage() -> Storage*& { return impl_view_.storage_; }

  auto groups_ptr() const -> uint8_t* { return impl_view_.groups_ptr(); }
  auto keys_ptr() const -> KeyT* { return impl_view_.keys_ptr(); }

  auto is_small() const -> bool { return size() <= small_size(); }
  auto small_size() const -> ssize_t {
    return static_cast<unsigned>(small_size_);
  }

  void Init(ssize_t init_size, Storage* init_storage);

  template <typename LookupKeyT>
  auto InsertIntoEmptyIndex(LookupKeyT lookup_key)
      -> std::pair<ssize_t, uint8_t>;

  template <typename LookupKeyT>
  auto EraseKey(LookupKeyT lookup_key) -> ssize_t;

  ViewBaseT impl_view_;
  int growth_budget_;
  int small_size_;
};

template <typename InputKeyT, typename InputValueT>
class RawHashtableBase : protected RawHashtableKeyBase<InputKeyT> {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using KeyBaseT = RawHashtableKeyBase<InputKeyT>;

  static constexpr bool HasValue = !std::is_same_v<ValueT, void>;

  // We have an important optimization for trivially relocatable keys. But we
  // don't have a relocatable trait (yet) so we approximate it here.
  static constexpr bool IsKeyTriviallyRelocatable =
      std::is_trivially_move_constructible_v<KeyT> &&
      std::is_trivially_destructible_v<KeyT>;

  static constexpr bool IsValueTriviallyRelocatable =
      HasValue && std::is_trivially_move_constructible_v<ValueT> &&
      std::is_trivially_destructible_v<ValueT>;

  static constexpr auto ComputeStorageSize(ssize_t size) -> ssize_t {
    if constexpr (!HasValue) {
      return ComputeKeyStorageSize<KeyT>(size);
    } else {
      return ComputeKeyValueStorageSize<KeyT, ValueT>(size);
    }
  }

  static constexpr auto ComputeStorageAlignment() -> std::align_val_t {
    if constexpr (!HasValue) {
      return std::align_val_t(StorageAlignment<KeyT>);
    } else {
      return std::align_val_t(StorageAlignment<KeyT, ValueT>);
    }
  }

  static auto Allocate(ssize_t size) -> RawHashtable::Storage* {
    return reinterpret_cast<RawHashtable::Storage*>(__builtin_operator_new(
        ComputeStorageSize(size), ComputeStorageAlignment(), std::nothrow_t()));
  }

  static auto Deallocate(Storage* storage, ssize_t size) -> void {
    ssize_t allocated_size = ComputeStorageSize(size);
    // We don't need the size, but make sure it always compiles.
    (void)allocated_size;
    return __builtin_operator_delete(storage,
#if __cpp_sized_deallocation
                                     allocated_size,
#endif
                                     ComputeStorageAlignment());
  }

  RawHashtableBase(int small_size, RawHashtable::Storage* small_storage)
      : KeyBaseT(small_size, small_storage) {}

  ~RawHashtableBase();

  auto values_ptr() const -> ValueT* {
    return reinterpret_cast<ValueT*>(
        reinterpret_cast<unsigned char*>(this->keys_ptr()) +
        ComputeValueStorageOffset<KeyT, ValueT>(this->size()));
  }

  template <typename LookupKeyT>
  auto GrowRehashAndInsertIndex(LookupKeyT lookup_key)
      -> std::pair<ssize_t, uint8_t>;
  template <typename LookupKeyT>
  auto InsertIndexHashed(LookupKeyT lookup_key) -> std::pair<ssize_t, uint8_t>;

  auto ClearImpl() -> void;
  auto DestroyImpl() -> void;
};

inline auto ComputeProbeMaskFromSize(ssize_t size) -> size_t {
  CARBON_DCHECK(llvm::isPowerOf2_64(size))
      << "Size must be a power of two for a hashed buffer!";
  // The probe mask needs to mask down to keep the index within
  // `groups_size`. Since `groups_size` is a power of two, this is equivalent to
  // `groups_size - 1`.
  return (size - 1) & ~GroupMask;
}

// This class handles building a sequence of probe indices from a given
// starting point, including both the quadratic growth and masking the index
// to stay within the bucket array size. The starting point doesn't need to be
// clamped to the size ahead of time (or even by positive), we will do it
// internally.
//
// We compute the quadratic probe index incrementally, but we can also compute
// it mathematically and will check that the incremental result matches our
// mathematical expectation. We use the quadratic probing formula of:
//
//   p(x,s) = (x + (s + s^2)/2) mod (Size / GroupSize)
//
// This particular quadratic sequence will visit every value modulo the
// provided size divided by the group size.
//
// However, we compute it scaled to the group size constant G and have it visit
// each G multiple modulo the size using the scaled formula:
//
//   p(x,s) = (x + (s + (s * s * G)/(G * G))/2) mod Size
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
#ifndef NDEBUG
    CARBON_DCHECK(
        i ==
        ((Start +
          ((Step + (Step * Step * GroupSize) / (GroupSize * GroupSize)) / 2)) %
         Size))
        << "Index in probe sequence does not match the expected formula.";
    CARBON_DCHECK(Step < Size) << "We necessarily visit all groups, so we "
                                  "can't have more probe steps than groups.";
#endif
  }

  auto getIndex() const -> ssize_t { return i; }
};

inline auto ComputeSeed() -> uint64_t {
  return reinterpret_cast<uint64_t>(&global_addr_seed);
}

inline auto ComputeControlByte(size_t tag) -> uint8_t {
  // Mask one over the high bit so that engaged control bytes are easily
  // identified.
  return tag | 0b10000000;
}

// TODO: Evaluate keeping this outlined to see if macro benchmarks observe the
// same perf hit as micros.
template <typename KeyT>
template <typename LookupKeyT>
auto RawHashtableViewBase<KeyT>::LookupIndexHashed(LookupKeyT lookup_key) const
    -> ssize_t {
  ssize_t local_size = size();
  uint8_t* groups = groups_ptr();
  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>(local_size);
  uint8_t control_byte = ComputeControlByte(tag);
  // ssize_t hash_index = ComputeHashIndex(hash, groups);

  KeyT* keys = reinterpret_cast<KeyT*>(
      reinterpret_cast<unsigned char*>(groups) + local_size);
  ProbeSequence s(hash_index, local_size);
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

template <typename InputKeyT>
template <typename IndexCallbackT, typename GroupCallbackT>
[[clang::always_inline]] void RawHashtableViewBase<InputKeyT>::ForEachIndex(
    IndexCallbackT index_callback, GroupCallbackT group_callback) {
  uint8_t* groups = this->groups_ptr();
  KeyT* keys = this->keys_ptr();

  ssize_t local_size = this->size();
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += GroupSize) {
    auto g = Group::Load(groups, group_index);
    auto present_matched_range = g.MatchPresent();
    if (!present_matched_range) {
      continue;
    }
    for (ssize_t byte_index : present_matched_range) {
      ssize_t index = group_index + byte_index;
      index_callback(keys, index);
    }

    group_callback(groups, group_index);
  }
}

template <typename InputKeyT>
auto RawHashtableViewBase<InputKeyT>::CountProbedKeys() const -> ssize_t {
  uint8_t* groups = this->groups_ptr();
  KeyT* keys = this->keys_ptr();

  ssize_t local_size = this->size();
  ssize_t count = 0;
  for (ssize_t group_index = 0; group_index < local_size;
       group_index += GroupSize) {
    auto g = RawHashtable::Group::Load(groups, group_index);
    auto present_matched_range = g.MatchPresent();
    for (ssize_t byte_index : present_matched_range) {
      ssize_t index = group_index + byte_index;
      HashCode hash = HashValue(keys[index], ComputeSeed());
      ssize_t hash_index =
          hash.ExtractIndexAndTag<7>(local_size).first & ~GroupMask;
      count += static_cast<ssize_t>(hash_index != group_index);
    }
  }
  return count;
}

template <typename InputKeyT>
template <typename LookupKeyT>
[[clang::noinline]] auto RawHashtableKeyBase<InputKeyT>::InsertIntoEmptyIndex(
    LookupKeyT lookup_key) -> std::pair<ssize_t, uint8_t> {
  uint8_t* groups = groups_ptr();
  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>(size());
  uint8_t control_byte = ComputeControlByte(tag);

  for (ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = Group::Load(groups, group_index);

    if (auto empty_matched_range = g.MatchEmpty()) {
      ssize_t index = group_index + *empty_matched_range.begin();
      return {index, control_byte};
    }

    // Otherwise we continue probing.
  }
}

inline auto ComputeNewSize(ssize_t old_size) -> ssize_t {
  // We want the next power of two. This should always be a power of two coming
  // in, and so we just verify that. Also verify that this doesn't overflow.
  CARBON_DCHECK(old_size == (ssize_t)llvm::PowerOf2Ceil(old_size))
      << "Expected a power of two!";
  return old_size * 2;
}

inline auto GrowthThresholdForSize(ssize_t size) -> ssize_t {
  // We use a 7/8ths load factor to trigger growth.
  return size - size / 8;
}

template <typename InputKeyT>
void RawHashtableKeyBase<InputKeyT>::Init(ssize_t init_size,
                                          Storage* init_storage) {
  size() = init_size;
  storage() = init_storage;
  std::memset(groups_ptr(), 0, init_size);
  growth_budget_ = GrowthThresholdForSize(init_size);
}

template <typename InputKeyT>
template <typename LookupKeyT>
auto RawHashtableKeyBase<InputKeyT>::EraseKey(LookupKeyT lookup_key)
    -> ssize_t {
  ssize_t index = impl_view_.LookupIndexHashed(lookup_key);
  if (index < 0) {
    return index;
  }

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  uint8_t* groups = this->groups_ptr();
  ssize_t group_index = index & ~GroupMask;
  auto g = Group::Load(groups, group_index);
  auto empty_matched_range = g.MatchEmpty();
  if (empty_matched_range) {
    groups[index] = Group::Empty;
    ++this->growth_budget_;
  } else {
    groups[index] = Group::Deleted;
  }

  // Also destroy the key while we're here.
  KeyT* keys = this->keys_ptr();
  keys[index].~KeyT();

  return index;
}

template <typename InputKeyT, typename InputValueT>
RawHashtableBase<InputKeyT, InputValueT>::~RawHashtableBase() {
  DestroyImpl();
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto
RawHashtableBase<InputKeyT, InputValueT>::GrowRehashAndInsertIndex(
    LookupKeyT lookup_key) -> std::pair<ssize_t, uint8_t> {
  // We collect the probed elements in a small vector for re-insertion. It is
  // tempting to re-use the already allocated storage, but doing so appears to
  // be a (very slight) performance regression. These are relatively rare and
  // storing them into the existing storage creates stores to the same regions
  // of memory we're reading. Moreover, it requires moving both the key and the
  // value twice, and doing the `memcpy` widening for relocatable types before
  // the group walk rather than after the group walk. In practice, between the
  // statistical rareness and using a large small size buffer on the stack, we
  // can handle this most efficiently with temporary storage.
  llvm::SmallVector<ssize_t, 128> probed_indices;

  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  ssize_t old_size = this->size();
  CARBON_DCHECK(old_size > 0);
  CARBON_DCHECK(this->growth_budget_ == 0);

  bool old_small = this->is_small();
  Storage* old_storage = this->storage();
  uint8_t* old_groups = this->groups_ptr();
  KeyT* old_keys = this->keys_ptr();
  ValueT* old_values;
  if constexpr (HasValue) {
    old_values = this->values_ptr();
  }

#ifndef NDEBUG
  ssize_t debug_empty_count =
      llvm::count(llvm::ArrayRef(old_groups, old_size), Group::Empty) +
      llvm::count(llvm::ArrayRef(old_groups, old_size), Group::Deleted);
  CARBON_DCHECK(debug_empty_count >=
                (old_size - GrowthThresholdForSize(old_size)));
#endif

  // Compute the new size and grow the storage in place (if possible).
  ssize_t new_size = ComputeNewSize(old_size);
  this->size() = new_size;
  this->storage() = Allocate(new_size);
  this->growth_budget_ = GrowthThresholdForSize(new_size);

  // Now extract the new components of the table.
  uint8_t* new_groups = this->groups_ptr();
  KeyT* new_keys = this->keys_ptr();
  ValueT* new_values;
  if constexpr (HasValue) {
    new_values = this->values_ptr();
  }

  // The common case, especially for large sizes, is that we double the size
  // when we grow. This allows an important optimization -- we're adding
  // exactly one more high bit to the hash-computed index for each entry. This
  // in turn means we can classify every entry in the table into three cases:
  //
  // 1) The new high bit is zero, the entry is at the same index in the new
  //    table as the old.
  //
  // 2) The new high bit is one, the entry is at the old index plus the old
  //    size.
  //
  // 3) The entry's current index doesn't match the initial hash index because
  //    it required some amount of probing to find an empty slot.
  //
  // The design of the hash table is specifically to minimize how many entries
  // fall into case (3), so we expect the vast majority of entries to be in
  // (1) or (2). This lets us model growth notionally as duplicating the hash
  // table up by `old_size` bytes, clearing out the empty slots, and inserting
  // any probed elements.

  ssize_t count = 0;
  for (ssize_t group_index = 0; group_index < old_size;
       group_index += GroupSize) {
    auto g = RawHashtable::Group::Load(old_groups, group_index);
    g.ClearDeleted();
    g.Store(new_groups, group_index);
    g.Store(new_groups, group_index | old_size);
    auto present_matched_range = g.MatchPresent();
    for (ssize_t byte_index : present_matched_range) {
      ++count;
      ssize_t old_index = group_index + byte_index;
      CARBON_DCHECK(new_groups[old_index] == old_groups[old_index]);
      CARBON_DCHECK(new_groups[old_index | old_size] == old_groups[old_index]);
      HashCode hash = HashValue(old_keys[old_index], ComputeSeed());
      ssize_t old_hash_index =
          hash.ExtractIndexAndTag<7>(old_size).first & ~GroupMask;
      if (LLVM_UNLIKELY(old_hash_index != group_index)) {
        probed_indices.push_back(old_index);
        new_groups[old_index] = Group::Empty;
        new_groups[old_index | old_size] = Group::Empty;
        continue;
      }
      ssize_t new_index =
          hash.ExtractIndexAndTag<7>(new_size).first & ~GroupMask;
      CARBON_DCHECK(new_index == old_hash_index ||
                    new_index == (old_hash_index | old_size));
      new_index += byte_index;
      // Toggle the newly added bit of the index to get to the other possible
      // target index.
      new_groups[new_index ^ old_size] = Group::Empty;

      // If we need to explicitly move (and destroy) the key or value, do so
      // here where we already know its target.
      if constexpr (!IsKeyTriviallyRelocatable) {
        new (&new_keys[new_index]) KeyT(std::move(old_keys[old_index]));
        old_keys[old_index].~KeyT();
      }
      if constexpr (HasValue && !IsValueTriviallyRelocatable) {
        new (&new_values[new_index]) ValueT(std::move(old_values[old_index]));
        old_values[old_index].~ValueT();
      }
    }
  }
  CARBON_DCHECK((count - static_cast<ssize_t>(probed_indices.size())) ==
                (new_size - llvm::count(llvm::ArrayRef(new_groups, new_size),
                                        Group::Empty)));
#ifndef NDEBUG
  CARBON_DCHECK(debug_empty_count == (old_size - count));
  CARBON_DCHECK(
      llvm::count(llvm::ArrayRef(new_groups, new_size), Group::Empty) ==
      debug_empty_count + static_cast<ssize_t>(probed_indices.size()) +
          old_size);
#endif

  // If the keys or values are trivially relocatable, we do a bulk memcpy of
  // them into place. This will copy them into both possible locations, which is
  // fine. One will be empty and clobbered if reused or ignored. The other will
  // be the one used. This might seem like it needs it to be valid for us to
  // create two copies, but it doesn't. This produces the exact same storage as
  // copying the storage into the wrong location first, and then again into the
  // correct location. Only one is live and only one is destroyed.
  if constexpr (IsKeyTriviallyRelocatable) {
    memcpy(new_keys, old_keys, old_size * sizeof(KeyT));
    memcpy(new_keys + old_size, old_keys, old_size * sizeof(KeyT));
  }
  if constexpr (IsValueTriviallyRelocatable) {
    memcpy(new_values, old_values, old_size * sizeof(ValueT));
    memcpy(new_values + old_size, old_values, old_size * sizeof(ValueT));
  }

  // We have to use the normal insert for anything that was probed before, but
  // we know we'll find an empty slot, so leverage that. We extract the probed
  // keys from the bottom of the old keys storage.
  for (ssize_t old_index : probed_indices) {
    KeyT old_key = std::move(old_keys[old_index]);
    old_keys[old_index].~KeyT();

    // We may end up needing to do a sequence of re-inserts, swapping out keys
    // and values each time, so we enter a loop here and break out of it for the
    // simple cases of re-inserting into a genuinely empty slot.
    auto [new_index, control_byte] = this->InsertIntoEmptyIndex(old_key);
    new (&new_keys[new_index]) KeyT(std::move(old_key));

    if constexpr (HasValue) {
      if (new_index == old_index) {
        new (&new_values[new_index]) ValueT(std::move(old_values[old_index]));
        old_values[old_index].~ValueT();
      }
    }

    new_groups[new_index] = control_byte;
  }
  CARBON_DCHECK(count ==
                (new_size - llvm::count(llvm::ArrayRef(new_groups, new_size),
                                        Group::Empty)));
  this->growth_budget_ -= count;
  CARBON_DCHECK(this->growth_budget_ ==
                (GrowthThresholdForSize(new_size) -
                 (new_size - llvm::count(llvm::ArrayRef(new_groups, new_size),
                                         Group::Empty))));
  CARBON_DCHECK(this->growth_budget_ > 0 &&
                "Must still have a growth budget after rehash!");

  if (!old_small) {
    // Old isn't a small buffer, so we need to deallocate it.
    Deallocate(old_storage, old_size);
  }

  // And lastly insert the lookup_key into an index in the newly grown map and
  // return that index for use.
  --this->growth_budget_;
  return this->InsertIntoEmptyIndex(lookup_key);
}

// Tries to insert the given lookup key into the map. Returns three pieces of
// data compressed into two registers (in order to avoid an in-memory return).
// These are the group pointer, a bool representing whether insertion is in fact
// required, and the byte index of either the found entry in the group or the
// slot of the group to insert into. The index will be `-1` if insertion
// isn't possible without growing. Last but not least, because we leave this
// outlined for code size, we also need to encode the `bool` in a way that is
// effective with various encodings and ABIs. Currently this is `uint32_t` as
// that seems to result in good code.
template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto
RawHashtableBase<InputKeyT, InputValueT>::InsertIndexHashed(
    LookupKeyT lookup_key) -> std::pair<ssize_t, uint8_t> {
  if (LLVM_UNLIKELY(this->size() == 0)) {
    this->Init(MinAllocatedSize, Allocate(MinAllocatedSize));
    return this->InsertIntoEmptyIndex(lookup_key);
  }

  uint8_t* groups = this->groups_ptr();

  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>(this->size());
  uint8_t control_byte = ComputeControlByte(tag);

  // We re-purpose the empty control byte to signal no insert is needed to the
  // caller. This is guaranteed to not be a control byte we're inserting.
  constexpr uint8_t NoInsertNeeded = Group::Empty;

  ssize_t group_with_deleted_index = -1;
  Group::MatchedRange deleted_matched_range;

  auto return_insert_at_index =
      [&](ssize_t index) -> std::pair<uint32_t, ssize_t> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    return {index, control_byte};
  };

  for (ProbeSequence s(hash_index, this->size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = Group::Load(groups, group_index);

    auto control_byte_matched_range = g.Match(control_byte);
    auto empty_matched_range = g.MatchEmpty();
    if (control_byte_matched_range) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = group_index + *byte_it;
        if (LLVM_LIKELY(this->keys_ptr()[index] == lookup_key)) {
          return {index, NoInsertNeeded};
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
    if (!empty_matched_range) {
      continue;
    }
    // Ok, we've finished probing without finding anything and need to insert
    // instead.

    // If we found a deleted slot, we don't need the probe sequence to insert
    // so just bail. We want to ensure building up a table is fast so we
    // de-prioritize this a bit. In practice this doesn't have too much of an
    // effect.
    if (LLVM_UNLIKELY(group_with_deleted_index >= 0)) {
      return return_insert_at_index(group_with_deleted_index +
                                    *deleted_matched_range.begin());
    }

    // We're going to need to grow by inserting into an empty slot. Check that
    // we have the budget for that before we compute the exact index of the
    // empty slot. Without the growth budget we'll have to completely rehash and
    // so we can just bail here.
    if (LLVM_UNLIKELY(this->growth_budget_ == 0)) {
      return this->GrowRehashAndInsertIndex(lookup_key);
    }

    --this->growth_budget_;
    return return_insert_at_index(group_index + *empty_matched_range.begin());
  }

  CARBON_FATAL() << "We should never finish probing without finding the entry "
                    "or an empty slot.";
}

template <typename InputKeyT, typename InputValueT>
auto RawHashtableBase<InputKeyT, InputValueT>::ClearImpl() -> void {
  this->impl_view_.ForEachIndex(
      [this](KeyT* /*keys*/, ssize_t index) {
        this->keys_ptr()[index].~KeyT();
        if constexpr (HasValue) {
          this->values_ptr()[index].~ValueT();
        }
      },
      [](uint8_t* groups, ssize_t group_index) {
        // Clear the group.
        std::memset(groups + group_index, 0, GroupSize);
      });
  this->growth_budget_ = GrowthThresholdForSize(this->size());
}

template <typename InputKeyT, typename InputValueT>
auto RawHashtableBase<InputKeyT, InputValueT>::DestroyImpl() -> void {
  // Nothing to do when in the un-allocated and unused state.
  if (this->size() == 0) {
    return;
  }

  // Destroy all the keys and, if present, values.
  this->impl_view_.ForEachIndex(
      [this](KeyT* /*keys*/, ssize_t index) {
        this->keys_ptr()[index].~KeyT();
        if constexpr (HasValue) {
          this->values_ptr()[index].~ValueT();
        }
      },
      [](auto...) {});

  // If small, nothing to deallocate.
  if (this->is_small()) {
    return;
  }

  // Just deallocate the storage without updating anything when destroying the
  // object.
  Deallocate(this->storage(), this->size());
}

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_H_