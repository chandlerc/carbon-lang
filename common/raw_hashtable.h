// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_H_
#define CARBON_COMMON_RAW_HASHTABLE_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <iterator>
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

namespace Carbon::RawHashtable {

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

template <typename KeyT>
constexpr ssize_t StorageAlignment =
    std::max<ssize_t>({GroupSize, alignof(Group), alignof(KeyT)});

template <typename KeyT>
constexpr auto ComputeKeyStorageOffset(ssize_t size) -> ssize_t {
  // There are `size` control bytes plus any alignment needed for the key type.
  return llvm::alignTo<alignof(KeyT)>(size);
}

template <ssize_t MinSmallSize>
constexpr auto ComputeSmallSize() -> ssize_t {
  return llvm::alignTo<GroupSize>(MinSmallSize);
}

template <typename KeyT, ssize_t SmallSize>
struct SmallSizeStorage;

template <typename KeyT>
struct SmallSizeStorage<KeyT, 0> : Storage {
  SmallSizeStorage() {}
  union {
    KeyT keys[0];
  };
};

template <typename KeyT, ssize_t SmallSize>
struct alignas(StorageAlignment<KeyT>) SmallSizeStorage : Storage {
  SmallSizeStorage() {}

  // FIXME: One interesting question is whether the small size should be a
  // minimum here or an exact figure.
  static_assert(llvm::isPowerOf2_64(SmallSize),
                "SmallSize must be a power of two for a hashed buffer!");
  static_assert(SmallSize >= GroupSize,
                "SmallSize must be at least the size of one group!");
  static_assert((SmallSize % GroupSize) == 0,
                "SmallSize must be a multiple of the group size!");
  static constexpr ssize_t SmallNumGroups = SmallSize / GroupSize;
  static_assert(llvm::isPowerOf2_64(SmallNumGroups),
                "The number of groups must be a power of two when hashing!");

  Group groups[SmallNumGroups];

  union {
    KeyT keys[SmallSize];
  };
};

template <typename KeyT>
class RawHashtableBase;

template <typename InputKeyT>
class RawHashtableViewBase {
 public:
  using KeyT = InputKeyT;

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool;

 protected:
  friend class RawHashtableBase<KeyT>;

  RawHashtableViewBase() = default;
  RawHashtableViewBase(ssize_t size, Storage* storage)
      : size_(size), storage_(storage) {}

  auto size() const -> ssize_t { return size_; }

  auto groups_ptr() const -> uint8_t* {
    return reinterpret_cast<uint8_t*>(storage_);
  }
  auto keys_ptr() const -> KeyT* {
    CARBON_DCHECK(llvm::isPowerOf2_64(size()))
        << "Size must be a power of two for a hashed buffer!";
    CARBON_DCHECK(size() == ComputeKeyStorageOffset<KeyT>(size()))
        << "Cannot be more aligned than a power of two.";
    return reinterpret_cast<KeyT*>(reinterpret_cast<unsigned char*>(storage_) +
                                   size());
  }

  template <typename IndexCallbackT, typename GroupCallbackT>
  void ForEachIndex(IndexCallbackT index_callback,
                    GroupCallbackT group_callback);

  ssize_t size_;
  Storage* storage_;
};

template <typename InputKeyT>
class RawHashtableBase {
 public:
  using KeyT = InputKeyT;
  using ViewBaseT = RawHashtableViewBase<KeyT>;
  // using LookupResultT = SetInternal::LookupResult<KeyT>;
  // using InsertResultT = SetInternal::InsertResult<KeyT>;

  template <typename LookupKeyT>
  auto Contains(LookupKeyT lookup_key) const -> bool {
    return ViewBaseT(*this).Contains(lookup_key);
  }

 protected:
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

  // NOLINTNEXTLINE(google-explicit-constructor): Designed to implicitly decay.
  operator ViewBaseT() const { return impl_view_; }

  auto size() const -> ssize_t { return impl_view_.size_; }
  auto size() -> ssize_t& { return impl_view_.size_; }
  auto storage() const -> Storage* { return impl_view_.storage_; }
  auto storage() -> Storage*& { return impl_view_.storage_; }

  auto groups_ptr() -> uint8_t* { return impl_view_.groups_ptr(); }
  auto keys_ptr() -> KeyT* { return impl_view_.keys_ptr(); }

  auto is_small() const -> bool { return size() <= small_size(); }
  auto small_size() const -> ssize_t {
    return static_cast<unsigned>(small_size_);
  }

  void Init(ssize_t init_size, Storage* init_storage);

  template <typename LookupKeyT>
  auto InsertIndexHashed(LookupKeyT lookup_key) -> std::pair<uint32_t, ssize_t>;
  template <typename LookupKeyT>
  auto InsertIntoEmptyIndex(LookupKeyT lookup_key) -> ssize_t;

  template <typename LookupKeyT>
  auto EraseKey(LookupKeyT lookup_key) -> ssize_t;

  template <typename IndexCallback>
  auto ClearImpl(IndexCallback index_callback) -> void;

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
    CARBON_DCHECK(
        i ==
        ((Start +
          ((Step + (Step * Step * GroupSize) / (GroupSize * GroupSize)) / 2)) %
         Size))
        << "Index in probe sequence does not match the expected formula.";
    CARBON_DCHECK(Step < Size) << "We necessarily visit all groups, so we "
                                  "can't have more probe steps than groups.";
  }

  auto getIndex() const -> ssize_t { return i; }
};

inline auto ComputeControlByte(size_t tag) -> uint8_t {
  // Mask one over the high bit so that engaged control bytes are easily
  // identified.
  return tag | 0b10000000;
}

template <typename KeyT, typename LookupKeyT>
//[[clang::noinline]]
auto LookupIndexHashed(LookupKeyT lookup_key, ssize_t size, Storage* storage)
    -> ssize_t {
  uint8_t* groups = reinterpret_cast<uint8_t*>(storage);
  auto seed = reinterpret_cast<uint64_t>(groups);
  HashCode hash = HashValue(lookup_key, seed);
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>(size);
  uint8_t control_byte = ComputeControlByte(tag);
  // ssize_t hash_index = ComputeHashIndex(hash, groups);

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

template <typename InputKeyT>
template <typename LookupKeyT>
auto RawHashtableViewBase<InputKeyT>::Contains(LookupKeyT lookup_key) const
    -> bool {
  Prefetch(storage_);
  return LookupIndexHashed<KeyT>(lookup_key, size(), storage_) >= 0;
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

// Tries to insert the given lookup key into the map. Returns three pieces of
// data compressed into two registers (in order to avoid an in-memory return).
// These are the group pointer, a bool representing whether insertion is in fact
// required, and the byte index of either the found entry in the group or the
// slot of the group to insert into. The index will be `-1` if insertion
// isn't possible without growing. Last but not least, because we leave this
// outlined for code size, we also need to encode the `bool` in a way that is
// effective with various encodings and ABIs. Currently this is `uint32_t` as
// that seems to result in good code.
template <typename InputKeyT>
template <typename LookupKeyT>
[[clang::noinline]] auto RawHashtableBase<InputKeyT>::InsertIndexHashed(
    LookupKeyT lookup_key) -> std::pair<uint32_t, ssize_t> {
  uint8_t* groups = groups_ptr();

  auto seed = reinterpret_cast<uint64_t>(groups);
  HashCode hash = HashValue(lookup_key, seed);
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>(size());
  uint8_t control_byte = ComputeControlByte(tag);

  ssize_t group_with_deleted_index = -1;
  Group::MatchedRange deleted_matched_range;

  auto return_insert_at_index = [&](ssize_t index) -> std::pair<bool, ssize_t> {
    // We'll need to insert at this index so set the control group byte to the
    // proper value.
    groups[index] = control_byte;
    return {/*needs_insertion=*/true, index};
  };

  for (ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = Group::Load(groups, group_index);

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
    if (LLVM_LIKELY(!empty_matched_range)) {
      continue;
    }

    // Ok, we've finished probing without finding anything and need to insert
    // instead.
    if (LLVM_UNLIKELY(group_with_deleted_index >= 0)) {
      // If we found a deleted slot, we don't need the probe sequence to insert
      // so just bail.
      break;
    }

    // Otherwise, we're going to need to grow by inserting over one of these
    // empty slots. Check that we have the budget for that before we compute the
    // exact index of the empty slot. Without the growth budget we'll have to
    // completely rehash and so we can just bail here.
    if (LLVM_UNLIKELY(growth_budget_ == 0)) {
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

template <typename InputKeyT>
template <typename LookupKeyT>
[[clang::noinline]] auto RawHashtableBase<InputKeyT>::InsertIntoEmptyIndex(
    LookupKeyT lookup_key) -> ssize_t {
  uint8_t* groups = groups_ptr();
  auto seed = reinterpret_cast<uint64_t>(groups);
  HashCode hash = HashValue(lookup_key, seed);
  auto [hash_index, tag] = hash.ExtractIndexAndTag<7>(size());
  uint8_t control_byte = ComputeControlByte(tag);

  for (ProbeSequence s(hash_index, size());; s.step()) {
    ssize_t group_index = s.getIndex();
    auto g = Group::Load(groups, group_index);

    if (auto empty_matched_range = g.MatchEmpty()) {
      ssize_t index = group_index + *empty_matched_range.begin();
      groups[index] = control_byte;
      return index;
    }

    // Otherwise we continue probing.
  }
}

inline auto ComputeNewSize(ssize_t old_size) -> ssize_t {
  if (old_size < (4 * GroupSize)) {
    // If we're going to heap allocate, get at least four groups.
    return 4 * GroupSize;
  }

  // Otherwise, we want the next power of two. This should always be a power of
  // two coming in, and so we just verify that. Also verify that this doesn't
  // overflow.
  CARBON_DCHECK(old_size == (ssize_t)llvm::PowerOf2Ceil(old_size))
      << "Expected a power of two!";
  return old_size * 2;
}

inline auto GrowthThresholdForSize(ssize_t size) -> ssize_t {
  // We use a 7/8ths load factor to trigger growth.
  return size - size / 8;
}

template <typename InputKeyT>
void RawHashtableBase<InputKeyT>::Init(ssize_t init_size,
                                       Storage* init_storage) {
  size() = init_size;
  storage() = init_storage;
  std::memset(groups_ptr(), 0, init_size);
  growth_budget_ = GrowthThresholdForSize(init_size);
}

template <typename InputKeyT>
template <typename LookupKeyT>
auto RawHashtableBase<InputKeyT>::EraseKey(LookupKeyT lookup_key) -> ssize_t {
  ssize_t index = LookupIndexHashed<KeyT>(lookup_key, size(), storage());
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

template <typename InputKeyT>
template <typename IndexCallback>
void RawHashtableBase<InputKeyT>::ClearImpl(IndexCallback index_callback) {
  // Otherwise walk the non-empty slots in the control group destroying each
  // one and clearing out the group.
  this->impl_view_.ForEachIndex(
      index_callback, [](uint8_t* groups, ssize_t group_index) {
        // Clear the group.
        std::memset(groups + group_index, 0, GroupSize);
      });

  // And reset the growth budget.
  this->growth_budget_ = GrowthThresholdForSize(size());
}

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_H_
