// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_HASHING_H_
#define CARBON_COMMON_HASHING_H_

#include <algorithm>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>

#include "common/check.h"
#include "common/ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/PointerIntPair.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Allocator.h"
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

class HashCode : public Printable<HashCode> {
 public:
  HashCode() = default;

  explicit HashCode(uint64_t value) : value_(value) {}

  explicit operator uint64_t() const { return value_; }

  friend auto operator==(HashCode lhs, HashCode rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }
  friend auto operator!=(HashCode lhs, HashCode rhs) -> bool {
    return lhs.value_ != rhs.value_;
  }

  friend auto HashValue(HashCode code) -> HashCode { return code; }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x}", value_);
  }

 private:
  uint64_t value_ = 0;
};

// Computes a hash code for the provided value.
//
// This **not** a cryptographically secure or stable hash -- it is only designed
// for use with in-memory hash table style data structures. Be fast and
// effective for that use case is the guiding principle of its design.
//
// The specific hash codes returned are only stable within a single execution.
// Not only are they not guaranteed to be stable from execution to execution,
// the implementation will work hard to vary them as much as possible between
// executions.
//
// The core algorithm is directly based on the excellent AHash work done in
// Rust:
//   https://github.com/tkaitchuck/aHash
//
// Much like AHash, to the extent that it can resist hash-flooding DDoS style
// attacks, it does so, but with a priority on performance.
//
// This function supports many typical data types such as primitives, string-ish
// views, and types composing primitives transparently like pairs, tuples, and
// array-ish views. It is also extensible to support user-defined types.
//
// To add support for your type, you need to implement a customization point --
// a free function that can be found by ADL for your type -- called `CarbonHash`
// with the following signature:
//
// ```cpp
// auto CarbonHash(HashState& h, const YourType& value) -> void;
// ```
//
// The API of the `HashState` type can be used to incorporate data from your
// user-defined type. Note that you only need to include the data that would
// lead to an object of your type comparing equal or not-equal to some other
// object, including objects of other types if your type supports heterogeneous
// equality comparison.
//
// To illustrate this -- if your type has some fixed amount of data, maybe 32
// 4-byte integers, and equality comparison only operates on the *values* of
// those integers, then the size (32 integers, or 128 bytes) doesn't need to be
// included in the hash. It can't differ between objects. But if your type has a
// *dynamic* amount of data, and only a common prefix is compared or the sizes
// are compared as part of equality comparison, then that dynamic size *does*
// need to be included in the hash. There are specialized methods on `HashState`
// that reflect this difference.
//
// An interesting observation of this principle is that C-strings don't need to
// incorporate their length into their hash *if* they incorporate the null
// terminating byte in the hash -- because that terminating byte will be the
// difference in equality comparison. But if the C-strings can be compared with
// any kind of explicitly-sized string views, the size *does* need to be
// incorporated.
template <typename T>
inline auto HashValue(const T& value) -> HashCode;

// Computes a seeded hash code for the provided value.
//
// Most aspects of this function behave the same as the unseeded overload, see
// its documentation for an overview.
//
// The seed behavior allows further varying the computed hash code. For example,
// a data structure that can store a per-instance seed can compute per-instance
// hash codes. However, the seed itself needs to be of high quality to ensure
// the best performance with high hash quality, and so we require it to itself
// be a computed hash code.
template <typename T>
inline auto HashValue(const T& value, HashCode seed) -> HashCode;

// Accumulator for hash state that eventually produces a hash code.
//
// This type is primarily used by types to implement a customization point
// `CarbonHash` that will in turn be used by the `HashValue` function. See the
// `HashValue` function for a detailed overview.
//
// Only `HashValue` and the related hashing infrastructure should ever
// instantiate a state object. Customization points should simply use methods to
// update it.
class HashState {
public:
  inline auto Update(uint64_t data) -> void {
    buffer = FoldedMultiply(data ^ buffer, MulConstant);
  }

  inline auto UpdateTwoChunks(uint64_t data0, uint64_t data1) -> void  {
    uint64_t combined = FoldedMultiply(data0 ^ extra_keys[0], data1 ^ extra_keys[1]);
    buffer = llvm::rotl((buffer + pad) ^ combined, RotConstant);
  }
  
  auto UpdateByteSequence(llvm::ArrayRef<std::byte> bytes) -> void;

 private:
  template <typename T>
  friend auto HashValue(const T& value) -> HashCode;
  friend auto HashValue(llvm::StringRef s) -> HashCode;

  HashState() = default;
  explicit HashState(HashCode seed) : buffer(seed) {}

  inline auto ShortFinish() -> HashCode { return HashCode(buffer + pad); }

  inline auto Finish() -> HashCode {
    return HashCode(llvm::rotl(FoldedMultiply(buffer, pad), buffer & 63));
  }

  inline auto FoldedMultiply(uint64_t lhs, uint64_t rhs) -> uint64_t {
    // Use the C23 extended integer support that Clang provides as a general
    // language extension.
    using U128 = unsigned _BitInt(128);
    U128 result = static_cast<U128>(lhs) * static_cast<U128>(rhs);
    return static_cast<uint64_t>(result) ^ static_cast<uint64_t>(result >> 64);
  }

  // Random data taken from the hexadecimal digits of Pi's fractional component,
  // written in lexical order for convenience of reading. The resulting
  // byte-stream will be different due to little-endian integers. This can be
  // generated with the following shell script:
  //
  // ```sh
  // echo 'obase=16; scale=154; 4*a(1)' | env BC_LINE_LENGTH=132 bc -l \
  //  | cut -c 3- | tr '[:upper:]' '[:lower:]' \
  //  | sed -e "s/.\{4\}/&'/g" \
  //  | sed -e "s/\(.\{4\}'.\{4\}'.\{4\}'.\{4\}\)'/0x\1,\n/g"
  // ```
  static constexpr uint64_t RandomData[8] = {
      0x243f'6a88'85a3'08d3, 0x1319'8a2e'0370'7344, 0xa409'3822'299f'31d0,
      0x082e'fa98'ec4e'6c89, 0x4528'21e6'38d0'1377, 0xbe54'66cf'34e9'0c6c,
      0xc0ac'29b7'c97c'50dd, 0x3f84'd5b5'b547'0913,
  };

  // This constant from Knuth's PRNG. Claimed to work better than others from
  // "splitmix32" by the AHash authors.
  static constexpr uint64_t MulConstant = 6364136223846793005;

  // Undocumented constant from AHash for rotations.
  static constexpr uint64_t RotConstant = 23;

  uint64_t buffer = RandomData[0];
  uint64_t pad = RandomData[1];
  uint64_t extra_keys[2] = {RandomData[2], RandomData[3]};
};

namespace Detail {

inline auto CarbonHash(HashState &hash, int8_t value) -> void {
  hash.Update(static_cast<uint64_t>(value));
}

inline auto CarbonHash(HashState &hash, uint8_t value) -> void {
  hash.Update(static_cast<uint64_t>(value));
}

template <typename T>
inline auto CarbonHashDispatch(HashState &hash, const T& value) -> void {
  CarbonHash(hash, value);
}

}  // namespace Detail

// Compute a HashCode for any integer value.
//
// Note that this function is intended to compute the same HashCode for
// a particular value without regard to the pre-promotion type. This is in
// contrast to hash_combine which may produce different HashCodes for
// differing argument types even if they would implicit promote to a common
// type without changing the value.
//template <typename T,
//          typename = std::enable_if_t<std::is_integral<T>::value, void>>
//auto HashValue(T value) -> HashCode;

// Compute a HashCode for a pointer's address.
//
// Note that this hashes the *address*, and not the value nor the type.
//template <typename T> auto HashValue(const T *ptr) -> HashCode;

// Compute a HashCode for a pair of objects.
//template <typename T, typename U>
//auto HashValue(const std::pair<T, U> &arg) -> HashCode;

// Compute a HashCode for a tuple.
//template <typename... Ts>
//auto HashValue(const std::tuple<Ts...> &arg) -> HashCode;

// Compute a HashCode for a standard string.
//template <typename T>
//auto HashValue(const std::basic_string<T> &arg) -> HashCode;

// Compute a HashCode for a standard string.
//template <typename T> auto HashValue(const std::optional<T> &arg) -> HashCode;

template <typename T>
inline auto HashValue(const T& value) -> HashCode {
  HashState state;
  Detail::CarbonHashDispatch(state, value);
  return state.Finish();
}

inline auto HashValue(llvm::StringRef s) -> HashCode {
  HashState state;
  llvm::ArrayRef<std::byte> bytes(reinterpret_cast<const std::byte*>(s.data()),
                                  s.size());
  state.UpdateByteSequence(bytes);
  return state.Finish();
}

}  // namespace Carbon

#endif  // CARBON_COMMON_HASHING_H_
