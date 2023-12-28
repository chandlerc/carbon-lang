// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_H_
#define CARBON_COMMON_RAW_HASHTABLE_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
#include <numeric>
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
      index_ = static_cast<size_t>(llvm::countr_zero(mask_));
      if constexpr (Shift > 0) {
        // We need to shift the index. However, we ensure that only zeros are
        // shifted off here and leave an optimizer hint about that. The index
        // will often be scaled by the user of this and we want that scale to
        // fold with the right shift whenever it can. That means we need the
        // optimizer to know there weren't low one-bites being shifted off here.
        CARBON_DCHECK((index_ & ((static_cast<MaskT>(1) << Shift) - 1)) == 0);
        __builtin_assume((index_ & ((static_cast<MaskT>(1) << Shift) - 1)) ==
                         0);
        index_ >>= Shift;
      }
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

  using MatchRange = BitIndexRange<uint32_t, /*Shift=*/0,
                                   /*ZeroMask=*/0xFFFF0000>;

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

  auto Set(uint8_t byte, ssize_t index) -> void {
    std::memcpy(reinterpret_cast<std::byte*>(&byte_vec) + index, &byte, 1);
  }

  auto ClearDeleted() -> void {
    // We cat zero every byte that isn't present.
    byte_vec = _mm_blendv_epi8(_mm_setzero_si128(), byte_vec, byte_vec);
  }

  auto Match(uint8_t match_byte) const -> MatchRange {
    auto match_byte_vec = _mm_set1_epi8(match_byte);
    auto match_byte_cmp_vec = _mm_cmpeq_epi8(byte_vec, match_byte_vec);
    uint32_t mask = _mm_movemask_epi8(match_byte_cmp_vec);
    return MatchRange(mask);
  }

  auto MatchEmpty() const -> MatchRange { return Match(Empty); }

  auto MatchDeleted() const -> MatchRange { return Match(Deleted); }

  auto MatchPresent() const -> MatchRange {
    // We arrange the byte vector for present bytes so that we can directly
    // extract it as a mask.
    return MatchRange(_mm_movemask_epi8(byte_vec));
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
  static constexpr uint64_t LSBs = 0x0101'0101'0101'0101ULL;

  using MatchRange = BitIndexRange<uint64_t, /*Shift=*/3>;

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

  auto Set(uint8_t byte, ssize_t index) -> void {
    std::memcpy(reinterpret_cast<std::byte*>(&byte_vec) + index, &byte, 1);
  }

  auto Match(uint8_t match_byte) const -> MatchRange {
    auto match_byte_vec = vdup_n_u8(match_byte);
    auto match_byte_cmp_vec = vceq_u8(byte_vec, match_byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchRange(mask & LSBs);
  }

  auto MatchEmpty() const -> MatchRange {
    auto match_byte_cmp_vec = vceqz_u8(byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchRange(mask);
  }

  auto MatchDeleted() const -> MatchRange {
    auto match_byte_vec = vdup_n_u8(Deleted);
    auto match_byte_cmp_vec = vceq_u8(byte_vec, match_byte_vec);
    uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
    return MatchRange(mask);
  }

  auto MatchPresent() const -> MatchRange {
    // Just directly extract the bytes as the MSB already marks presence.
    uint64_t mask = vreinterpret_u64_u8(byte_vec)[0];
    return MatchRange((mask >> 7) & LSBs);
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

  auto Set(uint8_t byte, ssize_t index) -> void {
    std::memcpy(reinterpret_cast<std::byte*>(&group) + index, &byte, 1);
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

template <typename KeyT, typename ValueT>
struct StorageEntry {
  static constexpr bool IsTriviallyDestructible =
      std::is_trivially_destructible_v<KeyT> &&
      std::is_trivially_destructible_v<ValueT>;

  static constexpr bool IsTriviallyRelocatable =
      IsTriviallyDestructible && std::is_trivially_move_constructible_v<KeyT> &&
      std::is_trivially_move_constructible_v<ValueT>;

  // We handle destruction and move manually as we only want to expose distinct
  // `KeyT` and `ValueT` subobjects to user code that may need to do in-place
  // construction. As a consequence, this struct only provides the storage and
  // we have to manually manage the construction, move, and destruction of the
  // objects.
  auto Destroy() -> void {
    static_assert(!IsTriviallyDestructible,
                  "Should never instantiate when trivial!");
    key.~KeyT();
    value.~ValueT();
  }
  auto Move(StorageEntry& new_entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(&new_entry, this, sizeof(StorageEntry));
    } else {
      new (&new_entry.key) KeyT(std::move(key));
      key.~KeyT();
      new (&new_entry.value) KeyT(std::move(value));
      value.~ValueT();
    }
  }

  StorageEntry() {}

  union {
    KeyT key;
  };
  union {
    ValueT value;
  };
};

template <typename KeyT>
struct StorageEntry<KeyT, void> {
  static constexpr bool IsTriviallyDestructible =
      std::is_trivially_destructible_v<KeyT>;

  static constexpr bool IsTriviallyRelocatable =
      IsTriviallyDestructible && std::is_trivially_move_constructible_v<KeyT>;

  auto Destroy() -> void {
    static_assert(!IsTriviallyDestructible,
                  "Should never instantiate when trivial!");
    key.~KeyT();
  }
  auto Move(StorageEntry& new_entry) -> void {
    if constexpr (IsTriviallyRelocatable) {
      memcpy(&new_entry, this, sizeof(StorageEntry));
    } else {
      new (&new_entry.key) KeyT(std::move(key));
      key.~KeyT();
    }
  }

  StorageEntry() {}

  union {
    KeyT key;
  };
};

template <typename KeyT, typename ValueT>
struct StorageGroup {
  Group group;

  StorageEntry<KeyT, ValueT> entries[GroupSize];
};

template <typename KeyT, typename ValueT>
constexpr ssize_t StorageAlignment = std::max<ssize_t>(
    {GroupSize, alignof(Group), alignof(StorageEntry<KeyT, ValueT>)});

struct Storage {};

template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct SmallStorageImpl;

template <typename KeyT, typename ValueT>
struct alignas(StorageAlignment<KeyT, ValueT>) SmallStorageImpl<KeyT, ValueT, 0>
    : Storage {
  // SmallStorageImpl() = default;
  static constexpr ssize_t SmallNumGroups = 0;

  StorageGroup<KeyT, ValueT> groups[0];
};

// If allocating storage, allocate a minimum of one cacheline of group metadata
// and a minimum of one group.
constexpr ssize_t MinAllocatedSize =
    std::max<ssize_t>(64, MaxGroupSize) / GroupSize;

template <typename KeyT, typename ValueT>
constexpr static auto ComputeStorageSize(ssize_t size) -> ssize_t {
  return sizeof(StorageGroup<KeyT, ValueT>) * size;
}

template <typename KeyT, typename ValueT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT, ValueT>) SmallStorageImpl : Storage {
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

  SmallStorageImpl() {
    // Validate a collection of invariants between the small size storage layout
    // and the dynamically computed storage layout. We need the key type to be
    // complete so we do this in the constructor body.
    static_assert(SmallSize == 0 || alignof(SmallStorageImpl) ==
                                        StorageAlignment<KeyT, ValueT>,
                  "Small size buffer must have the same alignment as a heap "
                  "allocated buffer.");
    static_assert(
        SmallSize == 0 || sizeof(SmallStorageImpl) ==
                              ComputeStorageSize<KeyT, ValueT>(SmallNumGroups),
        "The small size storage needs to match the dynamically "
        "computed storage size.");
  }

  StorageGroup<KeyT, ValueT> groups[SmallNumGroups];
};

// Base class that encodes either the absence of a value or a value type.
template <typename KeyT, typename ValueT = void>
class RawHashtableBase;

template <typename InputKeyT, typename InputValueT = void>
class RawHashtableViewBase {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using EntryT = StorageEntry<KeyT, ValueT>;
  using GroupT = StorageGroup<KeyT, ValueT>;

  friend class RawHashtableBase<KeyT, ValueT>;

  RawHashtableViewBase() = default;
  RawHashtableViewBase(ssize_t size, Storage* storage)
      : size_(size), storage_(storage) {}

  auto size() const -> ssize_t { return size_; }

  auto groups() const -> GroupT* { return reinterpret_cast<GroupT*>(storage_); }
#if 0
  auto entries() const -> EntryT* {
    CARBON_DCHECK(size() == 0 || llvm::isPowerOf2_64(size()))
        << "Size must be a power of two for a hashed buffer!";
    CARBON_DCHECK(size() == ComputeStorageEntryOffset<KeyT, ValueT>(size()))
        << "Cannot be more aligned than a power of two.";
    return reinterpret_cast<EntryT*>(
        reinterpret_cast<unsigned char*>(storage_) + size());
  }
#endif

  template <typename LookupKeyT>
  auto LookupIndexHashed(LookupKeyT lookup_key) const
      -> std::pair<GroupT*, EntryT*>;

  template <typename IndexCallbackT, typename GroupCallbackT>
  void ForEachIndex(IndexCallbackT entry_callback,
                    GroupCallbackT group_callback);

  auto CountProbedKeys() const -> ssize_t;

  ssize_t size_;
  Storage* storage_;
};

template <typename InputKeyT, typename InputValueT>
class RawHashtableBase {
 protected:
  using KeyT = InputKeyT;
  using ValueT = InputValueT;
  using ViewBaseT = RawHashtableViewBase<KeyT, ValueT>;
  using EntryT = typename ViewBaseT::EntryT;
  using GroupT = typename ViewBaseT::GroupT;

  template <ssize_t SmallSize>
  using SmallStorageT = SmallStorageImpl<KeyT, ValueT, SmallSize>;

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
    return RawHashtable::ComputeStorageSize<KeyT, ValueT>(size);
  }

  static constexpr auto ComputeStorageAlignment() -> std::align_val_t {
    return std::align_val_t(StorageAlignment<KeyT, ValueT>);
  }

  static auto Allocate(ssize_t size) -> Storage* {
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

  RawHashtableBase(int small_size, Storage* small_storage) {
    Init(small_size, small_storage);
    small_size_ = small_size;
  }

  // An internal constructor used to build temporary map base objects with a
  // specific allocated size. This is used internally to build ephemeral maps.
  explicit RawHashtableBase(ssize_t arg_size, Storage* arg_storage) {
    Init(arg_size, arg_storage);
    small_size_ = 0;
  }

  ~RawHashtableBase();

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewBaseT() const { return impl_view_; }

  auto size() const -> ssize_t { return impl_view_.size_; }
  auto size() -> ssize_t& { return impl_view_.size_; }
  auto storage() const -> Storage* { return impl_view_.storage_; }
  auto storage() -> Storage*& { return impl_view_.storage_; }

  auto groups() const -> GroupT* { return impl_view_.groups(); }
  // auto entries() const -> EntryT* { return impl_view_.entries(); }

  auto is_small() const -> bool { return size() <= small_size(); }
  auto small_size() const -> ssize_t {
    return static_cast<unsigned>(small_size_);
  }

  void Init(ssize_t init_size, Storage* init_storage);

  template <typename LookupKeyT>
  auto InsertIntoEmptyIndex(LookupKeyT lookup_key)
      -> std::pair<GroupT*, EntryT*>;

  template <typename LookupKeyT>
  auto EraseKey(LookupKeyT lookup_key) -> std::pair<GroupT*, EntryT*>;

  template <typename LookupKeyT>
  auto GrowRehashAndInsertIndex(LookupKeyT lookup_key)
      -> std::pair<GroupT*, EntryT*>;
  template <typename LookupKeyT>
  auto InsertIndexHashed(LookupKeyT lookup_key) -> std::pair<GroupT*, EntryT*>;

  auto ClearImpl() -> void;
  auto DestroyImpl() -> void;

  ViewBaseT impl_view_;
  int growth_budget_;
  int small_size_;
};

inline auto ComputeProbeMaskFromSize(ssize_t size) -> size_t {
  CARBON_DCHECK(llvm::isPowerOf2_64(size))
      << "Size must be a power of two for a hashed buffer!";
  // The probe mask needs to mask down to keep the index within
  // `groups_size`. Since `groups_size` is a power of two, this is equivalent to
  // `groups_size - 1`.
  // return (size - 1) & ~GroupMask;
  return (size - 1);
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
    Step += 1;
    i = (i + Step) & Mask;
#ifndef NDEBUG
    CARBON_DCHECK(i ==
                  ((Start + ((Step + (Step * Step * 1) / (1 * 1)) / 2)) % Size))
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
template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto RawHashtableViewBase<InputKeyT, InputValueT>::LookupIndexHashed(
    LookupKeyT lookup_key) const -> std::pair<GroupT*, EntryT*> {
  ssize_t local_size = size();
  if (LLVM_UNLIKELY(local_size == 0)) {
    return {nullptr, nullptr};
  }
  GroupT* local_groups = groups();
  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  uint8_t control_byte = ComputeControlByte(tag);

  ProbeSequence s(hash_index, local_size);
  do {
    ssize_t group_index = s.getIndex();
    GroupT& g = local_groups[group_index];
    auto control_byte_matched_range = g.group.Match(control_byte);
    if (LLVM_LIKELY(control_byte_matched_range)) {
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        ssize_t index = *byte_it;
        EntryT* entry = &g.entries[index];
        if (LLVM_LIKELY(entry->key == lookup_key)) {
          return {&g, entry};
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // We failed to find a matching entry in this bucket, so check if there are
    // empty slots and we're done probing.
    auto empty_byte_matched_range = g.group.MatchEmpty();
    if (LLVM_LIKELY(empty_byte_matched_range)) {
      return {&g, nullptr};
    }

    s.step();
  } while (LLVM_UNLIKELY(true));
}

template <typename InputKeyT, typename InputValueT>
template <typename IndexCallbackT, typename GroupCallbackT>
[[clang::always_inline]] void
RawHashtableViewBase<InputKeyT, InputValueT>::ForEachIndex(
    IndexCallbackT entry_callback, GroupCallbackT group_callback) {
  GroupT* local_groups = groups();
  ssize_t local_size = size();
  for (GroupT& g : llvm::MutableArrayRef(local_groups, local_size)) {
    auto present_matched_range = g.group.MatchPresent();
    for (ssize_t byte_index : present_matched_range) {
      entry_callback(g.entries[byte_index]);
    }

    group_callback(g);
  }
}

template <typename InputKeyT, typename InputValueT>
auto RawHashtableViewBase<InputKeyT, InputValueT>::CountProbedKeys() const
    -> ssize_t {
  GroupT* local_groups = this->groups();
  ssize_t local_size = this->size();
  ssize_t count = 0;
  for (auto [group_index, g] :
       llvm::enumerate(llvm::ArrayRef(local_groups, local_size))) {
    auto present_matched_range = g.group.MatchPresent();
    for (ssize_t byte_index : present_matched_range) {
      HashCode hash = HashValue(g.entries[byte_index].key, ComputeSeed());
      ssize_t hash_index = hash.ExtractIndexAndTag<7>().first &
                           ComputeProbeMaskFromSize(local_size);
      count +=
          static_cast<ssize_t>(hash_index != static_cast<ssize_t>(group_index));
    }
  }
  return count;
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto
RawHashtableBase<InputKeyT, InputValueT>::InsertIntoEmptyIndex(
    LookupKeyT lookup_key) -> std::pair<GroupT*, EntryT*> {
  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  uint8_t control_byte = ComputeControlByte(tag);
  uint8_t* groups = groups_ptr();
  EntryT* local_entries = entries();

  GroupT* local_groups = groups();
  for (ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    GroupT& g = local_groups[group_index];

    if (auto empty_matched_range = g.group.MatchEmpty()) {
      ssize_t index = *empty_matched_range.begin();
      g.group.Set(control_byte, index);
      return {&g, &g.entries[index]};
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
  // We use a 7/8ths load factor to trigger growth. The size is also in groups
  // so scale it to entries for the growth threshold.
  size *= GroupSize;
  return size - size / 8;
}

template <typename InputKeyT, typename InputValueT>
void RawHashtableBase<InputKeyT, InputValueT>::Init(ssize_t init_size,
                                                    Storage* init_storage) {
  size() = init_size;
  storage() = init_storage;
  std::memset(groups(), 0, init_size * sizeof(GroupT));
  growth_budget_ = GrowthThresholdForSize(init_size);
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
auto RawHashtableBase<InputKeyT, InputValueT>::EraseKey(LookupKeyT lookup_key)
    -> std::pair<GroupT*, EntryT*> {
  auto [group, entry] = impl_view_.LookupIndexHashed(lookup_key);
  if (!entry) {
    return {group, entry};
  }
  ssize_t index = entry - &group->entries[0];

  // If there are empty slots in this group then nothing will probe past this
  // group looking for an entry so we can simply set this slot to empty as
  // well. However, if every slot in this group is full, it might be part of
  // a long probe chain that we can't disrupt. In that case we mark the group
  // as deleted to keep probes continuing past it.
  //
  // If we mark the slot as empty, we'll also need to increase the growth
  // budget.
  auto empty_matched_range = group->group.MatchEmpty();
  if (empty_matched_range) {
    group->group.Set(Group::Empty, index);
    ++this->growth_budget_;
  } else {
    group->group.Set(Group::Deleted, index);
  }

  if constexpr (!EntryT::IsTriviallyDestructible) {
    entry->Destroy();
    entry->Destroy();
  }

  return {group, entry};
}

template <typename InputKeyT, typename InputValueT>
RawHashtableBase<InputKeyT, InputValueT>::~RawHashtableBase() {
  DestroyImpl();
}

template <typename InputKeyT, typename InputValueT>
template <typename LookupKeyT>
[[clang::noinline]] auto
RawHashtableBase<InputKeyT, InputValueT>::GrowRehashAndInsertIndex(
    LookupKeyT lookup_key) -> std::pair<GroupT*, EntryT*> {
  // We collect the probed elements in a small vector for re-insertion. It is
  // tempting to reuse the already allocated storage, but doing so appears to
  // be a (very slight) performance regression. These are relatively rare and
  // storing them into the existing storage creates stores to the same regions
  // of memory we're reading. Moreover, it requires moving both the key and the
  // value twice, and doing the `memcpy` widening for relocatable types before
  // the group walk rather than after the group walk. In practice, between the
  // statistical rareness and using a large small size buffer on the stack, we
  // can handle this most efficiently with temporary storage.
  llvm::SmallVector<EntryT*, 128> probed_indices;

  // We grow into a new `MapBase` so that both the new and old maps are
  // fully functional until all the entries are moved over. However, we directly
  // manipulate the internals to short circuit many aspects of the growth.
  ssize_t old_size = this->size();
  CARBON_DCHECK(old_size > 0);
  CARBON_DCHECK(this->growth_budget_ == 0);

  bool old_small = this->is_small();
  Storage* old_storage = this->storage();
  llvm::MutableArrayRef<GroupT> old_groups(this->groups(), old_size);

#ifndef NDEBUG
  ssize_t debug_empty_count = std::accumulate(
      old_groups.begin(), old_groups.end(), 0,
      [](ssize_t init, const GroupT& g) {
        auto empty_match = g.group.MatchEmpty();
        auto deleted_match = g.group.MatchDeleted();
        return init + std::distance(empty_match.begin(), empty_match.end()) +
               std::distance(deleted_match.begin(), deleted_match.end());
      });
  CARBON_DCHECK(debug_empty_count >=
                (old_size * GroupSize - GrowthThresholdForSize(old_size)))
      << "debug_empty_count: " << debug_empty_count << ", size: " << old_size;
#endif

  // Compute the new size and grow the storage in place (if possible).
  ssize_t new_size = ComputeNewSize(old_size);
  this->size() = new_size;
  this->storage() = Allocate(new_size);
  this->growth_budget_ = GrowthThresholdForSize(new_size);

  // Now extract the new components of the table.
  llvm::MutableArrayRef<GroupT> new_groups(this->groups(), new_size);

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
  for (ssize_t group_index = 0; group_index < old_size; ++group_index) {
    GroupT& old_g = old_groups[group_index];
    Group metadata = old_g.group;
    GroupT& new_g_a = new_groups[group_index];
    GroupT& new_g_b = new_groups[group_index | old_size];
    metadata.ClearDeleted();
    new_g_a.group = metadata;
    new_g_b.group = metadata;
    auto present_matched_range = metadata.MatchPresent();
    for (ssize_t byte_index : present_matched_range) {
      ++count;
      HashCode hash = HashValue(old_g.entries[byte_index].key, ComputeSeed());
      ssize_t old_hash_index = hash.ExtractIndexAndTag<7>().first &
                               ComputeProbeMaskFromSize(old_size);
      if (LLVM_UNLIKELY(old_hash_index != group_index)) {
        probed_indices.push_back(&old_g.entries[byte_index]);
        new_g_a.group.Set(Group::Empty, byte_index);
        new_g_b.group.Set(Group::Empty, byte_index);
        continue;
      }
      ssize_t new_index = hash.ExtractIndexAndTag<7>().first &
                          ComputeProbeMaskFromSize(new_size);
      ssize_t new_index = hash.ExtractIndexAndTag<7>().first &
                          ComputeProbeMaskFromSize(new_size);
      CARBON_DCHECK(new_index == old_hash_index ||
                    new_index == (old_hash_index | old_size));
      // new_index += byte_index;
      //  Toggle the newly added bit of the index to get to the other possible
      //  target index.
      new_groups[new_index ^ old_size].group.Set(Group::Empty, byte_index);

      // If we need to explicitly move (and destroy) the key or value, do so
      // here where we already know its target.
      if constexpr (!EntryT::IsTriviallyRelocatable) {
        old_g.entries[byte_index].Move(
            new_groups[new_index].entries[byte_index]);
      }
    }

    // If the keys or values are trivially relocatable, we do a bulk memcpy of
    // them into place. This will copy them into both possible locations, which
    // is fine. One will be empty and clobbered if reused or ignored. The other
    // will be the one used. This might seem like it needs it to be valid for us
    // to create two copies, but it doesn't. This produces the exact same
    // storage as copying the storage into the wrong location first, and then
    // again into the correct location. Only one is live and only one is
    // destroyed.
    if constexpr (EntryT::IsTriviallyRelocatable) {
      memcpy(new_g_a.entries, old_g.entries, GroupSize * sizeof(EntryT));
      memcpy(new_g_b.entries, old_g.entries, GroupSize * sizeof(EntryT));
    }
  }
#ifndef NDEBUG
  ssize_t new_empty_count = std::accumulate(
      new_groups.begin(), new_groups.end(), 0,
      [](ssize_t init, const GroupT& g) {
        auto empty_match = g.group.MatchEmpty();
        return init + std::distance(empty_match.begin(), empty_match.end());
      });
  CARBON_DCHECK((count - static_cast<ssize_t>(probed_indices.size())) ==
                (new_size * GroupSize - new_empty_count));
  CARBON_DCHECK(debug_empty_count == (old_size * GroupSize - count));
  CARBON_DCHECK(new_empty_count ==
                debug_empty_count +
                    static_cast<ssize_t>(probed_indices.size()) +
                    old_size * GroupSize);
#endif

  // We have to use the normal insert for anything that was probed before, but
  // we know we'll find an empty slot, so leverage that. We extract the probed
  // keys from the bottom of the old keys storage.
  for (EntryT* old_entry : probed_indices) {
    auto [group, new_entry] = this->InsertIntoEmptyIndex(old_entry->key);
    old_entry->Move(*new_entry);
  }
#ifndef NDEBUG
  new_empty_count = std::accumulate(
      new_groups.begin(), new_groups.end(), 0,
      [](ssize_t init, const GroupT& g) {
        auto empty_match = g.group.MatchEmpty();
        return init + std::distance(empty_match.begin(), empty_match.end());
      });
  CARBON_DCHECK(count == (new_size * GroupSize - new_empty_count));
#endif
  this->growth_budget_ -= count;
#ifndef NDEBUG
  CARBON_DCHECK(this->growth_budget_ ==
                (GrowthThresholdForSize(new_size) -
                 (new_size * GroupSize - new_empty_count)));
#endif
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
//[[clang::noinline]]
auto RawHashtableBase<InputKeyT, InputValueT>::InsertIndexHashed(
    LookupKeyT lookup_key) -> std::pair<GroupT*, EntryT*> {
  if (LLVM_UNLIKELY(this->size() == 0)) {
    this->Init(MinAllocatedSize, Allocate(MinAllocatedSize));
    --this->growth_budget_;
    return {this->InsertIntoEmptyIndex(lookup_key), true};
  }

  GroupT* local_groups = this->groups();

  HashCode hash = HashValue(lookup_key, ComputeSeed());
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>();
  uint8_t control_byte = ComputeControlByte(tag);

  GroupT* group_with_deleted = nullptr;
  Group::MatchRange deleted_matched_range;

  auto return_insert_at_index =
      [&](GroupT* g, ssize_t index) -> std::pair<GroupT*, EntryT*> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    g->group.Set(control_byte, index);
    return {g, &g->entries[index]};
  };

  for (ProbeSequence s(hash_index, this->size());; s.step()) {
    ssize_t group_index = s.getIndex();
    GroupT& g = local_groups[group_index];

    auto control_byte_matched_range = g.group.Match(control_byte);
    auto empty_matched_range = g.group.MatchEmpty();
    if (control_byte_matched_range) {
      EntryT* group_entries = &local_entries[group_index];
      auto byte_it = control_byte_matched_range.begin();
      auto byte_end = control_byte_matched_range.end();
      do {
        EntryT& entry = g.entries[*byte_it];

        if (LLVM_LIKELY(entry.key == lookup_key)) {
          return {nullptr, &entry};
        }
        ++byte_it;
      } while (LLVM_UNLIKELY(byte_it != byte_end));
    }

    // Track the first group with a deleted entry that we could insert over.
    if (!group_with_deleted) {
      deleted_matched_range = g.group.MatchDeleted();
      if (deleted_matched_range) {
        group_with_deleted = &g;
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
    if (LLVM_UNLIKELY(group_with_deleted)) {
      return return_insert_at_index(group_with_deleted,
                                    *deleted_matched_range.begin());
    }

    // We're going to need to grow by inserting into an empty slot. Check that
    // we have the budget for that before we compute the exact index of the
    // empty slot. Without the growth budget we'll have to completely rehash and
    // so we can just bail here.
    if (LLVM_UNLIKELY(this->growth_budget_ == 0)) {
      return {this->GrowRehashAndInsertIndex(lookup_key), true};
    }

    --this->growth_budget_;
    return return_insert_at_index(&g, *empty_matched_range.begin());
  }

  CARBON_FATAL() << "We should never finish probing without finding the entry "
                    "or an empty slot.";
}

template <typename InputKeyT, typename InputValueT>
auto RawHashtableBase<InputKeyT, InputValueT>::ClearImpl() -> void {
  this->impl_view_.ForEachIndex(
      [](EntryT& entry) {
        if constexpr (!EntryT::IsTriviallyDestructible) {
          entry.Destroy();
        }
      },
      [](GroupT& g) {
        // Clear the group.
        std::memset(&g.group, Group::Empty, GroupSize);
      });
  this->growth_budget_ = GrowthThresholdForSize(this->size());
}

template <typename InputKeyT, typename InputValueT>
auto RawHashtableBase<InputKeyT, InputValueT>::DestroyImpl() -> void {
  // Nothing to do when in the un-allocated and unused state.
  if (this->size() == 0) {
    return;
  }

  // Destroy all the entries.
  if constexpr (!EntryT::IsTriviallyDestructible) {
    this->impl_view_.ForEachIndex([](EntryT& entry) { entry.Destroy(); },
                                  [](auto...) {});
  }

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
