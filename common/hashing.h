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
  HashState(HashState&& arg) = default;
  HashState(const HashState& arg) = delete;
  auto operator=(HashState&& rhs) -> HashState& = default;

  static auto HashOne(HashState hash, uint64_t data) -> HashState;
  static auto HashTwo(HashState hash, uint64_t data0, uint64_t data1)
      -> HashState;
  static auto HashSizedBytes(HashState hash, llvm::ArrayRef<std::byte> bytes)
      -> HashState;
  static auto HashSize(HashState hash, size_t size) -> HashState;

  template <typename T, typename = std::enable_if_t<
                            std::has_unique_object_representations_v<T>>>
  static auto Hash(HashState hash, const T& value) -> HashState;

  template <
      typename T, typename U,
      typename = std::enable_if_t<std::has_unique_object_representations_v<T> &&
                                  std::has_unique_object_representations_v<U>>>
  static auto Hash(HashState hash, const std::pair<T, U>& value) -> HashState;

  template <typename... Ts,
            typename = std::enable_if_t<
                (... && std::has_unique_object_representations_v<Ts>)>>
  static auto Hash(HashState hash, const std::tuple<Ts...>& value) -> HashState;

#if 0
  inline auto ShortFinish() -> HashCode {
    return HashCode(llvm::rotl(buffer, RandomData[2] & 63));
  }

  inline auto Finish() -> HashCode {
    return HashCode(llvm::rotl(FoldedMultiply(buffer, RandomData[2]), buffer & 63));
  }
#endif

  explicit operator HashCode() const { return HashCode(buffer); }

 private:
  template <typename T>
  friend auto HashValue(const T& value, HashCode seed) -> HashCode;
  template <typename T>
  friend auto HashValue(const T& value) -> HashCode;
  friend auto HashValue(llvm::StringRef s, HashCode seed) -> HashCode;

  static auto Read1(const std::byte* data) -> uint64_t;
  static auto Read2(const std::byte* data) -> uint64_t;
  static auto Read4(const std::byte* data) -> uint64_t;
  static auto Read8(const std::byte* data) -> uint64_t;
  static auto Read1To3(const std::byte* data, ssize_t size) -> uint64_t;
  static auto Read4To8(const std::byte* data, ssize_t size) -> uint64_t;
  static auto Read8To16(const std::byte* data, ssize_t size)
      -> std::pair<uint64_t, uint64_t>;

  template <typename T, typename = std::enable_if_t<
                            std::has_unique_object_representations_v<T>>>
  static auto ReadSmall(const T& value) -> uint64_t;

  static auto FoldedMultiply(uint64_t lhs, uint64_t rhs) -> uint64_t;

  static auto ComputeRandomData() -> std::array<uint64_t, 4>;

  inline static auto HashTwoImpl(HashState hash, uint64_t data0, uint64_t data1,
                                 llvm::ArrayRef<uint64_t> keys) -> HashState;

  explicit HashState(HashCode seed)
      : buffer(static_cast<uint64_t>(seed)) {}

  HashState() = default;

  // Random data that will be initialized on program start. This will vary as
  // much as possible from execution to execution, but should be stable when
  // debugging or using ptrace (anything that fully stabilizes ASLR).
  static const std::array<uint64_t, 4> RandomData;

  // An empty global variable with linkage whose address is used.
  static volatile char global_variable;

  // This constant from Knuth's PRNG. Claimed to work better than others from
  // "splitmix32" by the AHash authors.
  //static constexpr uint64_t MulConstant = 0x5851'f42d'4c95'7f2dU;
  // Actually use the constant from `fmix64` in MurmurHash3 as that seems to
  // work better.
  //static constexpr uint64_t MulConstant = 0xc4ce'b9fe'1a85'ec53U;
  static constexpr uint64_t MulConstant = 0xff51'afd7'ed55'8ccdU;

  // Undocumented constant from AHash for rotations.
  static constexpr uint64_t RotConstant = 23;

  uint64_t buffer;
};

namespace Detail {

inline auto CarbonHash(HashState hash, llvm::ArrayRef<std::byte> bytes) -> HashState {
  //hash.UpdateSize(bytes.size());
  hash = HashState::HashSizedBytes(std::move(hash), bytes);
  return hash;
}

inline auto CarbonHash(HashState hash, llvm::StringRef value) -> HashState {
  return CarbonHash(std::move(hash), llvm::ArrayRef<std::byte>(
                              reinterpret_cast<const std::byte*>(value.data()),
                              value.size()));
}

template <typename T, typename U,
          typename = std::enable_if_t<
              std::has_unique_object_representations_v<T> &&
              std::has_unique_object_representations_v<U> &&
              sizeof(T) <= sizeof(uint64_t) && sizeof(U) <= sizeof(uint64_t)>>
inline auto CarbonHash(HashState hash, const std::pair<T, U>& value)
    -> HashState {
  return HashState::Hash(std::move(hash), value);
}

static_assert(std::has_unique_object_representations_v<int8_t>);
static_assert(std::has_unique_object_representations_v<int16_t>);
static_assert(std::has_unique_object_representations_v<int32_t>);
static_assert(std::has_unique_object_representations_v<int64_t>);

template <typename T, typename = std::enable_if_t<
                          std::has_unique_object_representations_v<T>>>
inline auto CarbonHash(HashState hash, const T& value)
    -> HashState {
  return HashState::Hash(std::move(hash), value);
}

template <typename T>
inline auto CarbonHashDispatch(HashState hash, const T& value) -> HashState {
  return CarbonHash(std::move(hash), value);
}

}  // namespace Detail

template <typename T>
inline auto HashValue(const T& value, HashCode seed) -> HashCode {
  return static_cast<HashCode>(
      Detail::CarbonHashDispatch(HashState(seed), value));
}

template <typename T>
inline auto HashValue(const T& value) -> HashCode {
  return HashValue(value, HashCode(HashState::RandomData[0]));
}

inline auto HashState::Read1(const std::byte *data) -> uint64_t {
  uint8_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto HashState::Read2(const std::byte *data) -> uint64_t {
  uint16_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto HashState::Read4(const std::byte *data) -> uint64_t {
  uint32_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto HashState::Read8(const std::byte *data) -> uint64_t {
  uint64_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto HashState::Read1To3(const std::byte *data, ssize_t size) -> uint64_t {
  // Use carefully crafted indexing to avoid branches on the exact size while
  // reading.
  uint64_t byte0 = static_cast<uint8_t>(data[0]);
  uint64_t byte1 = static_cast<uint8_t>(data[size / 2]);
  uint64_t byte2 = static_cast<uint8_t>(data[size - 1]);
  return byte0 | (byte1 << ((size / 2) * 8)) | (byte2 << ((size - 1) * 8));
}

inline auto HashState::Read4To8(const std::byte *data, ssize_t size) -> uint64_t {
  uint32_t low;
  std::memcpy(&low, data, sizeof(low));
  uint32_t high;
  std::memcpy(&high, data + size - sizeof(high), sizeof(high));
  return low | (static_cast<uint64_t>(high) << ((size - sizeof(high)) * 8));
}

inline auto HashState::Read8To16(const std::byte* data, ssize_t size)
    -> std::pair<uint64_t, uint64_t> {
  uint64_t low;
  std::memcpy(&low, data, sizeof(low));
  uint64_t high;
  std::memcpy(&high, data + size - sizeof(high), sizeof(high));
  return {low, high};
}

inline auto HashState::FoldedMultiply(uint64_t lhs, uint64_t rhs) -> uint64_t {
  // Use the C23 extended integer support that Clang provides as a general
  // language extension.
  using U128 = unsigned _BitInt(128);
  U128 result = static_cast<U128>(lhs) * static_cast<U128>(rhs);
  return static_cast<uint64_t>(result) ^ static_cast<uint64_t>(result >> 64);
}

inline auto HashState::HashOne(HashState hash, uint64_t data) -> HashState {
  hash.buffer = FoldedMultiply(data ^ hash.buffer, MulConstant);
  return hash;
}

inline auto HashState::HashTwoImpl(HashState hash, uint64_t data0, uint64_t data1,
                                           llvm::ArrayRef<uint64_t> keys)
    -> HashState {
  uint64_t combined = FoldedMultiply(data0 ^ keys[2], data1 ^ keys[3]);
  // Uses `shld` with LLVM which is frustratingly slow. Is the rotation needed?
  hash.buffer = llvm::rotl((hash.buffer + keys[1]) ^ combined, RotConstant);
  //buffer = (buffer + keys[1]) ^ combined;
  return hash;
}

inline auto HashState::HashTwo(HashState hash, uint64_t data0, uint64_t data1)
    -> HashState {
  hash = HashTwoImpl(std::move(hash), data0, data1, RandomData);
  return hash;
}

  // We have special knowledge of sizes of in-memory allocations we are hashing:
  // the high 32-bits are largely unimportant. Buffers over 4GB in size are rare
  // to begin with, and also so slow to hash and hash so much data that the
  // mixture of size isn't important. That means the only important sizes to
  // effectively mix into the hash are small ones, and we can do that more
  // cheaply than the full folded multiply by just using the low 64-bits of the
  // result.
inline auto HashState::HashSize(HashState hash, size_t size) -> HashState {
  hash.buffer ^= (size + RandomData[2]) * MulConstant;
  // buffer *= MulConstant;
  return hash;
}

template <typename T, typename /*enable_if*/>
inline auto HashState::ReadSmall(const T& value) -> uint64_t {
  const auto* storage = reinterpret_cast<const std::byte*>(&value);
  if constexpr (sizeof(T) == 1) {
    return Read1(storage);
  } else if constexpr (sizeof(T) == 2) {
    return  Read2(storage);
  } else if constexpr (sizeof(T) == 3) {
    return Read2(storage) | (Read1(&storage[2]) << 16);
  } else if constexpr (sizeof(T) == 4) {
    return  Read4(storage);
  } else if constexpr (sizeof(T) == 5) {
    return  Read4(storage) | (Read1(&storage[4]) << 32);
  } else if constexpr (sizeof(T) == 6 || sizeof(T) == 7) {
    // Use overlapping 4-byte reads for 6 and 7 bytes.
    return  Read4(storage) | (Read4(&storage[sizeof(T) - 4]) << 32);
  } else if constexpr (sizeof(T) == 8) {
    return  Read8(storage);
  } else {
    static_assert(sizeof(T) <= 8);
  }
}

template <typename T, typename /*enable_if*/>
inline auto HashState::Hash(HashState hash, const T& value) -> HashState {
  // We don't need the size to be part of the hash, as the size here is just a
  // function of the type and we're hashing to distinguish different values of
  // the same type. So we just dispatch to the fastest path for the specific size in question.
  if constexpr (sizeof(T) <= 8) {
    return HashOne(std::move(hash), ReadSmall(value));
  }
  
  const auto* storage = reinterpret_cast<const std::byte*>(&value);
  if constexpr (8 < sizeof(T) && sizeof(T) <= 16) {
    auto values = Read8To16(storage, sizeof(T));
    return HashTwo(std::move(hash), values.first, values.second);
  }

  // Hashing the size isn't relevant here, but is harmless, so fall back to a
  // common code path.
  return HashSizedBytes(std::move(hash),
                        llvm::ArrayRef<std::byte>(storage, sizeof(T)));
}

template <typename T, typename U, typename /*enable_if*/>
inline auto HashState::Hash(HashState hash, const std::pair<T, U>& value)
    -> HashState {
  if constexpr (sizeof(T) <= 8 && sizeof(U) <= 8) {
    return HashTwo(std::move(hash), ReadSmall(value.first), ReadSmall(value.second));
  } else {
    const auto* storage0 = reinterpret_cast<const std::byte*>(&value.first);
    const auto* storage1 = reinterpret_cast<const std::byte*>(&value.second); 
    return HashSizedBytes(
        HashSizedBytes(std::move(hash),
                       llvm::ArrayRef<std::byte>(storage0, sizeof(T))),
        llvm::ArrayRef<std::byte>(storage1, sizeof(U)));
  }
}

#if 0
template <typename... Ts, typename /*enable_if*/>
inline auto HashState::Hash(HashState hash, const std::tuple<Ts...>& value)
    -> HashState {
  
}
#endif

}  // namespace Carbon

#endif  // CARBON_COMMON_HASHING_H_
