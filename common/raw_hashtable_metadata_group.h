// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_
#define CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_

#include <cstddef>
#include <cstring>
#include <iterator>

#include "common/check.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/bit.h"
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

// We define a constant max group size. The particular group size used in
// practice may vary, but we want to have some upper bound that can be reliably
// used when allocating memory to ensure we don't create fractional groups and
// memory allocation is done consistently across the architectures.
constexpr ssize_t MaxGroupSize = 16;

// An index encoded as low zero bits ending in (at least) one set high bit. The
// index can be extracted by counting the low zero bits. It's presence can be
// tested directly however by checking for any zero bits. The underlying type to
// be used is provided as `MaskT` which must be an unsigned integer type.
//
// The index can be encoded by a power-of-two multiple of zero bits (including
// 1), which we model as a _shift_ of the count of zero bits to produce the
// index. The encoding must be all zero bits and an exact power of two to ensure
// this shift doesn't round the count -- we want the shift to fold with any
// subsequent index shifts that are common for users of these indices.
//
// Last but not least, some bits of the underlying value may be known-zero,
// which can optimize various operations. These can be represented as a
// `ZeroMask`.
template <typename MaskT, int Shift = 0, MaskT ZeroMask = 0>
class BitIndex : public Printable<BitIndex<MaskT, Shift, ZeroMask>> {
 public:
  BitIndex() = default;
  explicit BitIndex(MaskT mask) : mask_(mask) {}

  friend auto operator==(BitIndex lhs, BitIndex rhs) -> bool {
    if (lhs.empty() || rhs.empty()) {
      return lhs.empty() == rhs.empty();
    }
    // For non-empty bit indices, only the number of low zero bits matters.
    return llvm::countr_zero(lhs.mask_) == llvm::countr_zero(rhs.mask_);
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x}", mask_);
  }

  explicit operator bool() const { return !empty(); }
  auto empty() const -> bool {
    CARBON_DCHECK((mask_ & ZeroMask) == 0) << "Unexpected non-zero bits!";
    __builtin_assume((mask_ & ZeroMask) == 0);
    return mask_ == 0;
  }

  auto index() -> ssize_t {
    CARBON_DCHECK(mask_ != 0) << "Cannot get an index from a zero mask!";
    __builtin_assume(mask_ != 0);
    ssize_t index = static_cast<size_t>(llvm::countr_zero(mask_));
    if constexpr (Shift > 0) {
      // We need to shift the index. However, we ensure that only zeros are
      // shifted off here and leave an optimizer hint about that. The index
      // will often be scaled by the user of this and we want that scale to
      // fold with the right shift whenever it can. That means we need the
      // optimizer to know there weren't low one-bites being shifted off here.
      CARBON_DCHECK((index & ((static_cast<MaskT>(1) << Shift) - 1)) == 0);
      __builtin_assume((index & ((static_cast<MaskT>(1) << Shift) - 1)) == 0);
      index >>= Shift;
    }
    return index;
  }

 private:
  MaskT mask_ = 0;
};

template <typename MaskT, int Shift = 0, MaskT ZeroMask = 0>
class BitIndexRange : public Printable<BitIndexRange<MaskT, Shift, ZeroMask>> {
  using BitIndexT = BitIndex<MaskT, Shift, ZeroMask>;

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
      index_ = BitIndexT(mask_).index();
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
  auto empty() const -> bool { return BitIndexT(mask_).empty(); }

  auto begin() const -> Iterator { return Iterator(mask_); }
  auto end() const -> Iterator { return Iterator(); }

  template <int N>
  auto Test() const -> bool {
    return mask_ & (static_cast<MaskT>(1) << N);
  }

  friend auto operator==(BitIndexRange lhs, BitIndexRange rhs) -> bool {
    return lhs.mask_ == rhs.mask_;
  }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x}", mask_);
  }

  explicit operator MaskT() const { return mask_; }
  explicit operator BitIndexT() const { return BitIndexT(mask_); }

 private:
  MaskT mask_ = 0;
};

// A group of metadata bytes that can be manipulated together.
//
// The metadata bytes used Carbon's hashtable implementation are designed to
// support manipulating as groups, either using architecture specific SIMD code
// sequences or using portable SIMD-in-an-integer-register code sequences. These
// operations are *extraordinarily* performance sensitive and in sometimes
// surprising ways. The implementations here are crafted specifically to
// optimize the particular usages in Carbon's hashtable and should not be
// expected to be reusable in any other context.
//
// Throughout the functions operating on this type we use the following pattern
// to have a fallback portable implementation that can be directly used in the
// absence of a SIMD implementation, but to have the *exact* code for that
// portable implementation also used to check that any SIMD implementation
// produces the same result as the portable one. This structure ensures we don't
// have any un-compiled or un-tested path through the portable code even on
// platforms where we use SIMD as we expect to practically only test on
// platforms with a SIMD implementation and so be at a high risk of bit-rot.
//
// ```cpp
// auto Operation(...) -> ... {
//   ... portable_result;
//   if constexpr (!UseSIMD || DebugChecks) {
//     portable_result = PortableCode(...);
//     if (!UseSIMD) {
//       return portable_result;
//     }
//   }
//   ... result;
// #if CARBON_USE_NEON_SIMD_CONTROL_GROUP
//   result = NeonCode(...);
// #elif CARBON_USE_X86_SIMD_CONTROL_GROUP
//   result = X86Code(...);
// #else
//   static_assert(!UseSIMD, "Unimplemented SIMD operation");
// #endif
//   CARBON_DCHECK(result == portable_result) << ...;
//   return result;
// }
// ```
class MetadataGroup : public Printable<MetadataGroup> {
 public:
  static constexpr ssize_t Size =
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
      8;
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
      16;
#else
      8;
#endif
  static_assert(Size >= 8);
  static_assert(Size % 8 == 0);
  static_assert(Size <= MaxGroupSize);
  static_assert(MaxGroupSize % Size == 0);
  static_assert(llvm::isPowerOf2_64(Size),
                "The group size must be a constant power of two so dividing by "
                "it is a simple shift.");
  static constexpr ssize_t Mask = Size - 1;

  // Each control byte can have special values. All special values have the
  // most significant bit set to distinguish them from the seven hash bits
  // stored when the control byte represents a full bucket.
  //
  // Otherwise, their values are chose primarily to provide efficient SIMD
  // implementations of the common operations on an entire control group.
  static constexpr uint8_t Empty = 0;
  static constexpr uint8_t Deleted = 1;

  static constexpr uint8_t PresentMask = 0b1000'0000;

  static constexpr uint64_t MSBs = 0x8080'8080'8080'8080ULL;
  static constexpr uint64_t LSBs = 0x0101'0101'0101'0101ULL;

  static constexpr bool FastByteClear = Size == 8;

  using MatchRange =
#if CARBON_USE_X86_SIMD_CONTROL_GROUP
      BitIndexRange<uint32_t, /*Shift=*/0, /*ZeroMask=*/0xFFFF0000>;
#else
      BitIndexRange<uint64_t, /*Shift=*/3>;
#endif

  using MatchIndex =
#if CARBON_USE_X86_SIMD_CONTROL_GROUP
      BitIndex<uint32_t, /*Shift=*/0, /*ZeroMask=*/0xFFFF0000>;
#else
      BitIndex<uint64_t, /*Shift=*/3>;
#endif

  union {
    uint8_t bytes[Size];
    uint64_t byte_ints[Size / 8];
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
    uint8x8_t byte_vec = {};
    static_assert(sizeof(byte_vec) == Size);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
    __m128i byte_vec = {};
    static_assert(sizeof(byte_vec) == Size);
#endif
  };

  auto Print(llvm::raw_ostream& out) const -> void;

  friend auto operator==(MetadataGroup lhs, MetadataGroup rhs) -> bool {
    return CompareEqual(lhs, rhs);
  }

  static auto Load(uint8_t* metadata, ssize_t index) -> MetadataGroup;
  auto Store(uint8_t* metadata, ssize_t index) const -> void;

  auto ClearByte(ssize_t byte_index) -> void;

  auto ClearDeleted() -> void;

  auto Match(uint8_t match_byte) const -> MatchRange;
  auto MatchPresent() const -> MatchRange;

  auto MatchEmpty() const -> MatchIndex;
  auto MatchDeleted() const -> MatchIndex;

  // private:
  static constexpr bool UseSIMD =
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP || CARBON_USE_X86_SIMD_CONTROL_GROUP
      true;
#else
      false;
#endif
  static constexpr bool DebugChecks =
#ifndef NDEBUG
      true;
#else
      false;
#endif

  static auto CompareEqual(MetadataGroup lhs, MetadataGroup rhs) -> bool;

  static auto PortableLoad(uint8_t* metadata, ssize_t index) -> MetadataGroup;
  auto PortableStore(uint8_t* metadata, ssize_t index) const -> void;

  auto PortableClearDeleted() -> void;

  auto PortableMatch(uint8_t match_byte) const -> MatchRange;
  auto PortableMatchPresent() const -> MatchRange;

  auto PortableMatchEmpty() const -> MatchIndex;
  auto PortableMatchDeleted() const -> MatchIndex;

#if CARBON_USE_X86_SIMD_CONTROL_GROUP
  auto X86SIMDMatch(uint8_t match_byte) const -> MatchRange;
#endif
};

// Promote the size and mask to top-level constants as we'll need to operate on
// the grouped structure outside of the metadata bytes.
constexpr ssize_t GroupSize = MetadataGroup::Size;
constexpr ssize_t GroupMask = MetadataGroup::Mask;

inline auto MetadataGroup::Load(uint8_t* metadata, ssize_t index)
    -> MetadataGroup {
  MetadataGroup portable_g;
  if constexpr (!UseSIMD || DebugChecks) {
    portable_g = PortableLoad(metadata, index);
    if constexpr (!UseSIMD) {
      return portable_g;
    }
  }
  MetadataGroup g;
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
  g.byte_vec = vld1_u8(metadata + index);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
  g.byte_vec = _mm_load_si128(reinterpret_cast<__m128i*>(metadata + index));
#endif
  CARBON_DCHECK(g == portable_g);
  return g;
}

inline auto MetadataGroup::Store(uint8_t* metadata, ssize_t index) const
    -> void {
  if constexpr (!UseSIMD) {
    std::memcpy(metadata + index, &bytes, Size);
    return;
  }
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
  vst1_u8(metadata + index, byte_vec);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
  _mm_store_si128(reinterpret_cast<__m128i*>(metadata + index), byte_vec);
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
#endif
  CARBON_DCHECK(0 == std::memcmp(metadata + index, &bytes, Size));
}

inline auto MetadataGroup::ClearByte(ssize_t byte_index) -> void {
  static_assert(FastByteClear, "Only use byte clearing when fast!");
  static_assert(Size == 8, "The clear implementation assumes an 8-byte group.");

  byte_ints[0] &= ~(static_cast<uint64_t>(0xff) << (byte_index * 8));
}

inline auto MetadataGroup::ClearDeleted() -> void {
  MetadataGroup portable_g = *this;
  if constexpr (!UseSIMD || DebugChecks) {
    portable_g.PortableClearDeleted();
    if constexpr (!UseSIMD) {
      *this = portable_g;
      return;
    }
  }
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
  byte_ints[0] &= (~LSBs | byte_ints[0] >> 7);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
  byte_vec = _mm_blendv_epi8(_mm_setzero_si128(), byte_vec, byte_vec);
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
#endif
  CARBON_DCHECK(*this == portable_g);
}

inline auto MetadataGroup::Match(uint8_t match_byte) const -> MatchRange {
  MatchRange portable_result;
  if constexpr (!UseSIMD || DebugChecks) {
    portable_result = PortableMatch(match_byte);
    if constexpr (!UseSIMD) {
      return portable_result;
    }
  }
  MatchRange result;
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
  auto match_byte_vec = vdup_n_u8(match_byte);
  auto match_byte_cmp_vec = vceq_u8(byte_vec, match_byte_vec);
  uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
  result = MatchRange(mask & LSBs);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
  result = X86SIMDMatch(match_byte);
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
#endif
  CARBON_DCHECK(result == portable_result)
      << "SIMD result '" << result << "' doesn't match portable result '"
      << portable_result << "'";
  return result;
}

inline auto MetadataGroup::MatchPresent() const -> MatchRange {
  MatchRange portable_result;
  if constexpr (!UseSIMD || DebugChecks) {
    portable_result = PortableMatchPresent();
    if constexpr (!UseSIMD) {
      return portable_result;
    }
  }
  MatchRange result;
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
  // Just directly extract the bytes as the MSB already marks presence.
  result = MatchRange((byte_ints[0] >> 7) & LSBs);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
  // We arrange the byte vector for present bytes so that we can directly
  // extract it as a mask.
  result = MatchRange(_mm_movemask_epi8(byte_vec));
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
#endif
  CARBON_DCHECK(result == portable_result)
      << "SIMD result '" << result << "' doesn't match portable result '"
      << portable_result << "'";
  return result;
}

inline auto MetadataGroup::MatchEmpty() const -> MatchIndex {
  MatchIndex portable_result;
  if constexpr (!UseSIMD || DebugChecks) {
    portable_result = PortableMatchEmpty();
    if constexpr (!UseSIMD) {
      return portable_result;
    }
  }
  MatchIndex result;
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
  auto match_byte_cmp_vec = vceqz_u8(byte_vec);
  uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
  result = MatchIndex(mask);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
  result = MatchIndex(X86SIMDMatch(Empty));
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
#endif
  CARBON_DCHECK(result == portable_result)
      << "SIMD result '" << result << "' doesn't match portable result '"
      << portable_result << "'";
  return result;
}

inline auto MetadataGroup::MatchDeleted() const -> MatchIndex {
  MatchIndex portable_result;
  if constexpr (!UseSIMD || DebugChecks) {
    portable_result = PortableMatchDeleted();
    if constexpr (!UseSIMD) {
      return portable_result;
    }
  }
  MatchIndex result;
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
  auto match_byte_vec = vdup_n_u8(Deleted);
  auto match_byte_cmp_vec = vceq_u8(byte_vec, match_byte_vec);
  uint64_t mask = vreinterpret_u64_u8(match_byte_cmp_vec)[0];
  result = MatchIndex(mask);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
  result = MatchIndex(X86SIMDMatch(Deleted));
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
#endif
  CARBON_DCHECK(result == portable_result)
      << "SIMD result '" << result << "' doesn't match portable result '"
      << portable_result << "'";
  return result;
}

inline auto MetadataGroup::CompareEqual(MetadataGroup lhs, MetadataGroup rhs)
    -> bool {
  bool portable_result;
  if constexpr (!UseSIMD || DebugChecks) {
    portable_result = llvm::equal(lhs.bytes, rhs.bytes);
    if (!UseSIMD) {
      return portable_result;
    }
  }
  bool result;
#if CARBON_USE_NEON_SIMD_CONTROL_GROUP
  result = vreinterpret_u64_u8(vceq_u8(lhs.byte_vec, rhs.byte_vec))[0] ==
           static_cast<uint64_t>(-1LL);
#elif CARBON_USE_X86_SIMD_CONTROL_GROUP
#if __SSE4_2__
  result = _mm_testc_si128(_mm_cmpeq_epi8(lhs.byte_vec, rhs.byte_vec),
                           _mm_set1_epi8(0xff)) == 1;
#else
  result = _mm_movemask_epi8(_mm_cmpeq_epi8(lhs.byte_vec, rhs.byte_vec)) ==
           0x0000'ffffU;
#endif
#else
  static_assert(!UseSIMD, "Unimplemented SIMD operation");
#endif
  CARBON_DCHECK(result == portable_result);
  return result;
}

inline auto MetadataGroup::PortableLoad(uint8_t* metadata, ssize_t index)
    -> MetadataGroup {
  MetadataGroup g;
  static_assert(sizeof(g) == Size);
  std::memcpy(&g.bytes, metadata + index, Size);
  return g;
}

inline auto MetadataGroup::PortableStore(uint8_t* metadata, ssize_t index) const
    -> void {
  std::memcpy(metadata + index, &bytes, Size);
}

inline auto MetadataGroup::PortableClearDeleted() -> void {
  for (uint64_t& byte_int : byte_ints) {
    byte_int &= (~LSBs | byte_int >> 7);
  }
}

inline auto MetadataGroup::PortableMatch(uint8_t match_byte) const
    -> MatchRange {
  // Use a simple fallback approach for sizes beyond 8.
  // TODO: Instead of a silly fallback, we should generalize the below
  // algorithm for sizes above 8, even if to just exercise the same code on
  // more platforms.
  if constexpr (Size > 8) {
    static_assert(Size <= 32, "Sizes larger than 32 not yet supported!");
    uint32_t mask = 0;
    uint32_t bit = 1;
    for (ssize_t i : llvm::seq<ssize_t>(0, Size)) {
      if (bytes[i] == match_byte) {
        mask |= bit;
      }
      bit <<= 1;
    }
    return MatchRange(mask);
  }

  // Small group optimized matching only works on present bytes, not empty or
  // deleted.
  CARBON_DCHECK(match_byte & 0b1000'0000) << llvm::formatv("{0:x}", match_byte);

  // This algorithm only works for matching *present* bytes. We leverage the
  // set high bit in the present case as part of the algorithm. The whole
  // algorithm has a critical path height of 5 operations, and does 7
  // operations total:
  //
  //          group | MSBs    LSBs * match_byte
  //                 \            /
  //                 mask ^ pattern
  //                      |
  // group & MSBs    MSBs - mask
  //        \            /
  //    group_MSBs & mask
  //               |
  //          mask >> 7
  //
  // While it is superficially similar to the "find zero bytes in a word" bit
  // math trick, it is different because this is designed to
  // have no false positives and perfectly produce 0x01 for matching bytes and
  // 0x00 for non-matching bytes. This is do-able because we constrain to only
  // handle present matches which only require testing 7 bits and have a
  // particular layout.
  //
  // It is tempting to remove the last shift by targeting 0x80 bytes on matches,
  // but this makes the shift to scale the zero-bit-count to an index to be a
  // significant shift that cannot be folded with shifts to scale the index to a
  // larger object size.

  // Set the high bit of every byte to `1`. The match byte always has this bit
  // set as well, which ensures the xor below, in addition to zeroing the byte
  // that matches, also clears the high bit of every byte.
  uint64_t mask = byte_ints[0] | MSBs;
  // Broadcast the match byte to all bytes.
  uint64_t pattern = LSBs * match_byte;
  // Xor the broadcast pattern, making matched bytes become zero bytes.
  mask = mask ^ pattern;
  // Subtract the mask bytes from `0x80` bytes so that any non-zero mask byte
  // clears the high bit but zero leaves it intact.
  mask = MSBs - mask;
  // Mask down to the high bits, but only those in the original group.
  mask &= (byte_ints[0] & MSBs);
  // And shift to the low bit so that counting the low zero bits exactly
  // produces a shifted index for a match.
  mask >>= 7;
#ifndef NDEBUG
  for (ssize_t byte_index : llvm::seq<ssize_t>(0, Size)) {
    uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
    if (bytes[byte_index] == match_byte) {
      CARBON_DCHECK(byte == 0x01)
          << "Should just have the low bit set for a present byte, found: "
          << llvm::formatv("{0:x}", byte);
    } else {
      CARBON_DCHECK(byte == 0)
          << "Should have no bits set for an unmatched byte, found: "
          << llvm::formatv("{0:x}", byte);
    }
  }
#endif
  return MatchRange(mask);
}

inline auto MetadataGroup::PortableMatchPresent() const -> MatchRange {
  if constexpr (Size > 8) {
    static_assert(Size <= 32, "Sizes larger than 32 not yet supported!");
    uint32_t mask = 0;
    uint32_t bit = 1;
    for (ssize_t i : llvm::seq<ssize_t>(0, Size)) {
      if (bytes[i] & PresentMask) {
        mask |= bit;
      }
      bit <<= 1;
    }
    return MatchRange(mask);
  }

  uint64_t mask = (byte_ints[0] >> 7) & LSBs;
#ifndef NDEBUG
  for (ssize_t byte_index : llvm::seq<ssize_t>(0, Size)) {
    uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
    if (bytes[byte_index] & 0b1000'0000U) {
      CARBON_DCHECK(byte == 0x01)
          << "Should just have the low bit set for a present byte, found: "
          << llvm::formatv("{0:x}", byte);
    } else {
      CARBON_DCHECK(byte == 0)
          << "Should have no bits set for an unmatched byte in '" << *this
          << "', found: " << llvm::formatv("{0:x}", byte);
    }
  }
#endif
  return MatchRange(mask);
}

inline auto MetadataGroup::PortableMatchEmpty() const -> MatchIndex {
  if constexpr (Size > 8) {
    return static_cast<MatchIndex>(PortableMatch(Empty));
  }

  // Materialize the group into a word.
  uint64_t mask = (byte_ints[0] >> 7) | byte_ints[0];
  mask = ~mask & LSBs;
#ifndef NDEBUG
  for (ssize_t byte_index : llvm::seq<ssize_t>(0, Size)) {
    uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
    if (bytes[byte_index] == Empty) {
      CARBON_DCHECK((byte & 1) == 1)
          << "Should have the low bit set for a matched byte, found: "
          << llvm::formatv("{0:x}", byte);
      // Only the first match is needed so stop scanning once found.
      break;
    }

    CARBON_DCHECK(byte == 0)
        << "Should have no bits set for an unmatched byte in '" << *this
        << "', found '" << llvm::formatv("{0:x}", byte) << "' at index "
        << byte_index;
  }
#endif
  return MatchIndex(mask);
}

inline auto MetadataGroup::PortableMatchDeleted() const -> MatchIndex {
  if constexpr (Size > 8) {
    return static_cast<MatchIndex>(PortableMatch(Deleted));
  }

  // Materialize the group into a word.
  uint64_t mask = (byte_ints[0] >> 7) | ~byte_ints[0];
  mask = ~mask & LSBs;
#ifndef NDEBUG
  for (ssize_t byte_index : llvm::seq<ssize_t>(0, Size)) {
    uint8_t byte = (mask >> (byte_index * 8)) & 0xFF;
    if (bytes[byte_index] == Deleted) {
      CARBON_DCHECK((byte & 1) == 1)
          << "Should have the low bit set for a matched byte, found: "
          << llvm::formatv("{0:x}", byte);
      // Only the first match is needed so stop scanning once found.
      break;
    }

    CARBON_DCHECK(byte == 0)
        << "Should have no bits set for an unmatched byte in '" << *this
        << "', found '" << llvm::formatv("{0:x}", byte) << "' at index "
        << byte_index;
  }
#endif
  return MatchIndex(mask);
}

#if CARBON_USE_X86_SIMD_CONTROL_GROUP
inline auto MetadataGroup::X86SIMDMatch(uint8_t match_byte) const
    -> MatchRange {
  auto match_byte_vec = _mm_set1_epi8(match_byte);
  auto match_byte_cmp_vec = _mm_cmpeq_epi8(byte_vec, match_byte_vec);
  uint32_t mask = _mm_movemask_epi8(match_byte_cmp_vec);
  return MatchRange(mask);
}
#endif

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_METADATA_GROUP_H_
