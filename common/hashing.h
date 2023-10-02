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

  constexpr explicit HashCode(uint64_t value) : value_(value) {}

  constexpr explicit operator uint64_t() const { return value_; }

  friend constexpr auto operator==(HashCode lhs, HashCode rhs) -> bool {
    return lhs.value_ == rhs.value_;
  }
  friend constexpr auto operator!=(HashCode lhs, HashCode rhs) -> bool {
    return lhs.value_ != rhs.value_;
  }

  friend auto CarbonHashValue(HashCode code) -> HashCode { return code; }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x16}", value_);
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

namespace SMHasher {
auto HashTest(const void* key, int len, uint32_t seed, void* out) -> void;
auto SizedHashTest(const void* key, int len, uint32_t seed, void* out) -> void;
}  // namespace SMHasher

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

  explicit operator HashCode() const { return HashCode(buffer); }

  static auto DumpRandomData() -> void;

 private:
  template <typename T>
  friend auto HashValue(const T& value, HashCode seed) -> HashCode;
  template <typename T>
  friend auto HashValue(const T& value) -> HashCode;
  friend auto HashValue(llvm::StringRef s, HashCode seed) -> HashCode;
  friend auto SMHasher::HashTest(const void* key, int len, uint32_t seed,
                                 void* out) -> void;
  friend auto SMHasher::SizedHashTest(const void* key, int len, uint32_t seed,
                                      void* out) -> void;

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

  static auto Mix(uint64_t lhs, uint64_t rhs) -> uint64_t;

  static auto ComputeRandomData() -> std::array<uint64_t, 8>;

  static auto RotState(HashState hash) -> HashState;

  static auto HashSizedBytesLarge(HashState hash,
                                  llvm::ArrayRef<std::byte> bytes) -> HashState;

  explicit HashState(HashCode seed) : buffer(static_cast<uint64_t>(seed)) {}

  HashState() = default;

  // Random data that will be initialized on program start. This will vary as
  // much as possible from execution to execution, but should be stable when
  // debugging or using ptrace (anything that fully stabilizes ASLR).
  static const std::array<uint64_t, 16> StaticRandomData;
  static const std::array<uint64_t, 8> RandomData;

  // An empty global variable with linkage whose address is used.
  static volatile char global_variable;

  // The multiplicative hash constant from Knuth, derived from 2^64 / Phi.
  static constexpr uint64_t MulConstant = 0x9e37'79b9'7f4a'7c15U;

  // Undocumented constant from AHash for rotations.
  static constexpr uint64_t RotConstant = 23;

  uint64_t buffer;
};

namespace Detail {

inline auto CarbonHash(HashState hash, llvm::ArrayRef<std::byte> bytes)
    -> HashState {
  hash = HashState::HashSizedBytes(std::move(hash), bytes);
  return hash;
}

inline auto CarbonHash(HashState hash, llvm::StringRef value) -> HashState {
  return CarbonHash(
      std::move(hash),
      llvm::ArrayRef<std::byte>(
          reinterpret_cast<const std::byte*>(value.data()), value.size()));
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
inline auto CarbonHash(HashState hash, const T& value) -> HashState {
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

inline auto HashState::Read1(const std::byte* data) -> uint64_t {
  uint8_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto HashState::Read2(const std::byte* data) -> uint64_t {
  uint16_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto HashState::Read4(const std::byte* data) -> uint64_t {
  uint32_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto HashState::Read8(const std::byte* data) -> uint64_t {
  uint64_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto HashState::Read1To3(const std::byte* data, ssize_t size)
    -> uint64_t {
  // Use carefully crafted indexing to avoid branches on the exact size while
  // reading.
  uint64_t byte0 = static_cast<uint8_t>(data[0]);
  uint64_t byte1 = static_cast<uint8_t>(data[size - 1]);
  uint64_t byte2 = static_cast<uint8_t>(data[size / 2]);
  return byte0 | (byte1 << 16) | (byte2 << 8);
}

inline auto HashState::Read4To8(const std::byte* data, ssize_t size)
    -> uint64_t {
  uint32_t low;
  std::memcpy(&low, data, sizeof(low));
  uint32_t high;
  std::memcpy(&high, data + size - sizeof(high), sizeof(high));
  return low | (static_cast<uint64_t>(high) << 32);
}

inline auto HashState::Read8To16(const std::byte* data, ssize_t size)
    -> std::pair<uint64_t, uint64_t> {
  uint64_t low;
  std::memcpy(&low, data, sizeof(low));
  uint64_t high;
  std::memcpy(&high, data + size - sizeof(high), sizeof(high));
  return {low, high};
}

inline auto HashState::Mix(uint64_t lhs, uint64_t rhs) -> uint64_t {
  // Use the C23 extended integer support that Clang provides as a general
  // language extension.
  using U128 = unsigned _BitInt(128);
  U128 result = static_cast<U128>(lhs) * static_cast<U128>(rhs);
  return static_cast<uint64_t>(result & ~static_cast<uint64_t>(0)) ^
         static_cast<uint64_t>(result >> 64);
}

inline auto HashState::HashOne(HashState hash, uint64_t data) -> HashState {
  // When hashing exactly one 64-bit entity use the Phi-derived constant as this
  // is just multiplicative hashing. The initial buffer is mixed on input to
  // pipeline with materializing the constant.
  hash.buffer = Mix(data ^ hash.buffer, MulConstant);
  return hash;
}

inline auto HashState::HashTwo(HashState hash, uint64_t data0, uint64_t data1)
    -> HashState {
  // When hashing two chunks of data at the same time, we mask it with two
  // unstable keys to avoid any crafted inputs from creating collisions. This
  // isn't as strong as masking with the current buffer but still provides some
  // resistance to crafted inputs and matches what AHash uses. However, we don't
  // use *consecutive* keys to avoid a common compiler "optimization" of loading
  // both 64-bit chunks into a 128-bit vector and doing the XOR in the vector
  // unit. The latency of extracting the data afterward eclipses any benefit.
  // Callers will routinely have two consecutive data values here, but using
  // non-consecutive keys avoids any vectorization being tempting.
  //
  // Note that AHash adds more random data to the incoming buffer and applies a
  // rotation at the end. The addition doesn't seem to improve things much. The
  // rotation is very impactful in long chains of hashing, but for short ones
  // doesn't matter. This code separates the rotation out and lets chained code
  // insert the rotation where needed. This improves the latency of applications
  // of this routine where the rotate adds little value.
  //
  // We XOR both the incoming state and a random word over the first data. This
  // is done to pipeline with materializing the constants and is observed to
  // have better performance than XOR-ing after the mix.
#if 1
  hash.buffer = Mix(hash.buffer ^ data0 ^ StaticRandomData[1],
                    data1 ^ StaticRandomData[3]);
#else
  hash.buffer ^= Mix(data0 ^ StaticRandomData[1], data1 ^ StaticRandomData[3]);
#endif
  return hash;
}

inline auto HashState::RotState(HashState hash) -> HashState {
  // Rotating the buffer helps repeated hashing mix more of the state. The
  // rotation amount of `53` is arbitrarily chosen to match the amount found
  // useful in ML-based experiments on Abseil's hash function.
  hash.buffer = llvm::rotr(hash.buffer, 53);
  return hash;
}

inline auto HashState::HashSizedBytes(HashState hash,
                                      llvm::ArrayRef<std::byte> bytes)
    -> HashState {
  const std::byte* data_ptr = bytes.data();
  const ssize_t size = bytes.size();

  // First handle short sequences under 8 bytes.
  if (LLVM_UNLIKELY(size == 0)) {
    hash = HashOne(std::move(hash), 0);
    return hash;
  }
  if (size <= 8) {
    uint64_t data;
    if (size >= 4) {
      data = Read4To8(data_ptr, size);
    } else {
      data = Read1To3(data_ptr, size);
    }
    // We optimize for latency on short strings by hashing both the data and
    // size in a single multiply here. This results in a *statistically* weak
    // hash function. It would be improved by doing two rounds of multiplicative
    // hashing which is what many other modern multiplicative hashes do,
    // including Abseil and others:
    //
    // ```cpp
    // hash = HashOne(std::move(hash), data0);
    // hash = HashOne(std::move(hash), size);
    // ```
    //
    // We opt to make the same tradeoff here for small sized strings that both
    // this library and Abseil make for *fixed* size integers by using a weaker
    // single round of multiplicative hashing.
    __asm volatile("# LLVM-MCA-BEGIN 8b-sized-hash" ::: "memory");
#if 1
    // The primary goal with short string hashing is *latency*, and this routine
    // optimizes heavily for it over quality. The incoming buffer is XOR-ed on
    // the data to overlap the latency of loading a size-dependent constant for
    // the multiplication. Having 8 constants available and selecting them with
    // the size helps incorporate the size in an extremely low-latency fashion
    // without dramatic compromises of hash quality.
    hash.buffer = Mix(data ^ hash.buffer, StaticRandomData[size - 1]);
#elif 1
    hash.buffer ^= Mix(data ^ StaticRandomData[size - 1], MulConstant);
#else
    hash.buffer = Mix(data ^ hash.buffer, MulConstant) ^ RandomData[size - 1];
#endif
    __asm volatile("# LLVM-MCA-END" ::: "memory");
    return hash;
  }

  if (size <= 16) {
    __asm volatile("# LLVM-MCA-BEGIN 16b-sized-hash" ::: "memory");
    auto data = Read8To16(data_ptr, size);
    // Similar to the above, we optimize primarily for latency here. One complex
    // tradeoff is the working-set size. Above we only use a single cache line
    // to encode the size, but here we pull from a 128-byte table. However it
    // results in dramatically smaller code footprint and so may still be a net
    // benefit compared to other approaches of incorporating both the size and
    // 16-bytes of data into the result. A variation with larger code-size but
    // only using a 64-byte table is included for benchmarking.
#if 1
    hash.buffer = Mix(hash.buffer ^ data.first,
                      data.second ^ StaticRandomData[(size - 1)]);
#else
    hash.buffer =
        Mix(hash.buffer ^ StaticRandomData[(size - 1) >> 1] ^ data.first,
            data.second ^ StaticRandomData[(size - 1) & 0b11]);
#endif
    __asm volatile("# LLVM-MCA-END" ::: "memory");
    return hash;
  }

  if (size <= 32) {
    __asm volatile("# LLVM-MCA-BEGIN 32b-sized-hash" ::: "memory");
#if 1
    // Do two mixes of overlapping 16-byte ranges in parallel to minimize
    // latency. The mix also needn't be *that* good as we'll do another round of
    // mixing with the size.
    const std::byte* tail_16b_ptr = data_ptr + (size - 16);
    uint64_t m0 = Mix(Read8(data_ptr) ^ StaticRandomData[1],
                      Read8(data_ptr + 8) ^ hash.buffer);
    // Rotate one arm of the mix. See the comments in `RotState` for details,
    // this is done manually to only one side to let it pipeline better.
    m0 = llvm::rotr(m0, 53);
    uint64_t m1 = Mix(Read8(tail_16b_ptr) ^ StaticRandomData[3],
                      Read8(tail_16b_ptr + 8) ^ hash.buffer);
    hash.buffer = m0 ^ m1;
    hash = HashOne(std::move(hash), size);
#endif
    __asm volatile("# LLVM-MCA-END" ::: "memory");
    return hash;
  }

  return HashSizedBytesLarge(std::move(hash), bytes);
}

template <typename T, typename /*enable_if*/>
inline auto HashState::ReadSmall(const T& value) -> uint64_t {
  const auto* storage = reinterpret_cast<const std::byte*>(&value);
  if constexpr (sizeof(T) == 1) {
    return Read1(storage);
  } else if constexpr (sizeof(T) == 2) {
    return Read2(storage);
  } else if constexpr (sizeof(T) == 3) {
    return Read2(storage) | (Read1(&storage[2]) << 16);
  } else if constexpr (sizeof(T) == 4) {
    return Read4(storage);
  } else if constexpr (sizeof(T) == 5) {
    return Read4(storage) | (Read1(&storage[4]) << 32);
  } else if constexpr (sizeof(T) == 6 || sizeof(T) == 7) {
    // Use overlapping 4-byte reads for 6 and 7 bytes.
    return Read4(storage) | (Read4(&storage[sizeof(T) - 4]) << 32);
  } else if constexpr (sizeof(T) == 8) {
    return Read8(storage);
  } else {
    static_assert(sizeof(T) <= 8);
  }
}

template <typename T, typename /*enable_if*/>
inline auto HashState::Hash(HashState hash, const T& value) -> HashState {
  // We don't need the size to be part of the hash, as the size here is just a
  // function of the type and we're hashing to distinguish different values of
  // the same type. So we just dispatch to the fastest path for the specific
  // size in question.
  if constexpr (sizeof(T) <= 8) {
    //__asm volatile("# LLVM-MCA-BEGIN 8b-hash":::"memory");
    hash = HashOne(std::move(hash), ReadSmall(value));
    //__asm volatile("# LLVM-MCA-END":::"memory");
    return hash;
  }

  const auto* data_ptr = reinterpret_cast<const std::byte*>(&value);
  if constexpr (8 < sizeof(T) && sizeof(T) <= 16) {
    //__asm volatile("# LLVM-MCA-BEGIN 16b-hash":::"memory");
    auto values = Read8To16(data_ptr, sizeof(T));
    hash = HashTwo(std::move(hash), values.first, values.second);
    //__asm volatile("# LLVM-MCA-END":::"memory");
    return hash;
  }

  if constexpr (16 < sizeof(T) && sizeof(T) <= 32) {
    // Do two mixes of overlapping 16-byte ranges in parallel to minimize
    // latency.
    const std::byte* tail_16b_ptr = data_ptr + (sizeof(T) - 16);
    uint64_t m0 = Mix(Read8(data_ptr) ^ StaticRandomData[1],
                      Read8(data_ptr + 8) ^ hash.buffer);
    // Rotate one arm of the mix. See the comments in `RotState` for details,
    // this is done manually to only one side to let it pipeline better.
    m0 = llvm::rotr(m0, 53);
    uint64_t m1 = Mix(Read8(tail_16b_ptr) ^ StaticRandomData[3],
                      Read8(tail_16b_ptr + 8) ^ hash.buffer);
    hash.buffer = m0 ^ m1;
    return hash;
  }

  // Hashing the size isn't relevant here, but is harmless, so fall back to a
  // common code path.
  return HashSizedBytesLarge(std::move(hash),
                             llvm::ArrayRef<std::byte>(data_ptr, sizeof(T)));
}

template <typename T, typename U, typename /*enable_if*/>
inline auto HashState::Hash(HashState hash, const std::pair<T, U>& value)
    -> HashState {
  if constexpr (sizeof(T) <= 8 && sizeof(U) <= 8) {
    return HashTwo(std::move(hash), ReadSmall(value.first),
                   ReadSmall(value.second));
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
