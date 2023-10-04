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
#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

// A type wrapping a 64-bit hash code and provides a basic but limited API.
// Conversion to an actual integer is explicit to avoid accidental, unintended
// arithmetic.
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

  friend auto CarbonHash(HashCode code) -> HashCode { return code; }

  auto Print(llvm::raw_ostream& out) const -> void {
    out << llvm::formatv("{0:x16}", value_);
  }

 private:
  uint64_t value_ = 0;
};

// Computes a hash code for the provided value.
//
// This **not** a cryptographically secure or stable hash -- it is only designed
// for use with in-memory hash table style data structures. Being fast and
// effective for that use case is the guiding principle of its design.
//
// There is no guarantee that the values produced are stable from execution to
// execution. For speed and quality reasons, the implementation does not
// introduce any variance to defend against accidental dependencies. As a
// consequence, it is strongly encouraged to use the seeded variant with some
// seed that varies from execution to execution to avoid depending on specific
// values produced.
//
// The algorithm used is most heavily based on Abseil's hashing algorithm[1], with
// some additional ideas and inspiration from the fallback hashing algorithm in
// Rust's AHash[2]. However, there are also *significant* changes introduced
// here and several are novel.
//
// [1]: https://github.com/abseil/abseil-cpp/tree/master/absl/hash/internal
// [2]: https://github.com/tkaitchuck/aHash/wiki/AHash-fallback-algorithm
//
// This hash algorithm does *not* defend against hash flooding in any
// interesting way. While it can be viewed as "keyed" on the seed, no
// cryptanalysis or differential analysis has been performed to try and ensure
// the seed cannot be cancelled out through some clever input. In general, this
// function works to be *fast* for hash tables. If you need to defend against
// hash flooding, either directly use a data structure with strong worst-case
// guarantees (for example, Abseil's b-tree containers), or a hash table which
// detects catastrophic collisions and falls back to such a data structure.
//
// This hash function is heavily optimized for *latency* over *quality*. Modern
// hash tables designs can efficiently handle reasonable collision rates,
// including using extra bits form the hash to avoid all efficiency coming from
// the same low bits. Because of this, low-latency is significantly more
// important for performance than high-quality, and we exploit this heavily. The
// result is that the hash codes produced *do* have significant avalanche
// problems for small keys and should only be used in data structures that can
// handle this. The upside is that the latency for hashing integers, pointers,
// and small byte strings (up to 32-bytes) is exceptionally low, and essentially
// a small constant time instruction sequence. Especially for short strings,
// this function is often significantly faster even than Abseil's hash function
// or any other we are aware of. Longer byte strings are reasonably fast as
// well, competitive or better than Abseil's hash function.
//
// No exotic instruction set extensions are required, and the state used is
// small. It does rely on being able to get the low- and high-64-bit results of
// a 64-bit multiply efficiently.
//
// The function supports many typical data types such as primitives, string-ish
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
// a data structure can provide a per-instance seed. The seed doesn't need to be
// of any particular high quality, but a zero seed has bad effects in several
// places. Prefer the unseeded routine rather than providing a zero here.
template <typename T>
inline auto HashValue(const T& value, uint64_t seed) -> HashCode;

// Forward declare some test routines that are made friends and used when
// linking against the SMHasher test infrastructure.
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

  // Convenience method to hash an object by its object representation when that
  // is known to be valid. This is primarily useful for builtin and primitive
  // types.
  // 
  // This can be directly used for simple users combining some aggregation of
  // objects. However, when possible, prefer the tuple version below for
  // aggregating several primitive types into a hash.
  template <typename T, typename = std::enable_if_t<
                            std::has_unique_object_representations_v<T>>>
  static auto Hash(HashState hash, const T& value) -> HashState;

  // Convenience method to hash a variable number of objects whose object representation
  // when that is known to be valid. This is primarily useful for for builtin
  // and primitive type objects.
  //
  // There is no guaranteed correspondence between the behavior of a single call
  // with multiple parameters and multiple calls. This routine is also optimized
  // for handling relatively small numbers of objects. For hashing large
  // aggregations, consider some Merkle-tree decomposition or arranging for a
  // byte buffer that can be hashed as a single buffer.
  template <typename... Ts,
            typename = std::enable_if_t<
                (... && std::has_unique_object_representations_v<Ts>)>>
  static auto Hash(HashState hash, const Ts&... value) -> HashState;

  static auto HashOne(HashState hash, uint64_t data) -> HashState;
  static auto HashTwo(HashState hash, uint64_t data0, uint64_t data1)
      -> HashState;
  static auto HashSizedBytes(HashState hash, llvm::ArrayRef<std::byte> bytes)
      -> HashState;

  explicit operator HashCode() const { return HashCode(buffer); }

 private:
  template <typename T>
  friend auto HashValue(const T& value, uint64_t seed) -> HashCode;
  template <typename T>
  friend auto HashValue(const T& value) -> HashCode;
  
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

  static auto HashSizedBytesLarge(HashState hash,
                                  llvm::ArrayRef<std::byte> bytes) -> HashState;

  explicit HashState(uint64_t seed) : buffer(seed) {}

  HashState() = default;

  // The multiplicative hash constant from Knuth, derived from 2^64 / Phi.
  static constexpr uint64_t MulConstant = 0x9e37'79b9'7f4a'7c15U;

  // Random data taken from the hexadecimal digits of Pi's fractional component,
  // written in lexical order for convenience of reading. The resulting
  // byte-stream will be different due to little-endian integers. The
  // initializers here can be generated with the following shell script:
  //
  // ```sh
  // echo 'obase=16; scale=308; 4*a(1)' | env BC_LINE_LENGTH=500 bc -l \
  //  | cut -c 3- | tr '[:upper:]' '[:lower:]' \
  //  | sed -e "s/.\{4\}/&'/g" \
  //  | sed -e "s/\(.\{4\}'.\{4\}'.\{4\}'.\{4\}\)'/0x\1,\n/g"
  // ```
  static inline constexpr std::array<uint64_t, 16> StaticRandomData = {
      0x243f'6a88'85a3'08d3, 0x1319'8a2e'0370'7344, 0xa409'3822'299f'31d0,
      0x082e'fa98'ec4e'6c89, 0x4528'21e6'38d0'1377, 0xbe54'66cf'34e9'0c6c,
      0xc0ac'29b7'c97c'50dd, 0x3f84'd5b5'b547'0917, 0x9216'd5d9'8979'fb1b,
      0xd131'0ba6'98df'b5ac, 0x2ffd'72db'd01a'dfb7, 0xb8e1'afed'6a26'7e96,
      0xba7c'9045'f12c'7f99, 0x24a1'9947'b391'6cf7, 0x0801'f2e2'858e'fc16,
      0x6369'20d8'7157'4e68,
  };

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

inline auto CarbonHash(HashState hash, std::string_view value) -> HashState {
  return CarbonHash(
      std::move(hash),
      llvm::ArrayRef<std::byte>(
          reinterpret_cast<const std::byte*>(value.data()), value.size()));
}

inline auto CarbonHash(HashState hash, const std::string& value) -> HashState {
  return CarbonHash(
      std::move(hash),
      llvm::ArrayRef<std::byte>(
          reinterpret_cast<const std::byte*>(value.data()), value.size()));
}

// C++ guarantees this is true for the unsigned variants, but we require it for
// signed variants and pointers.
static_assert(std::has_unique_object_representations_v<int8_t>);
static_assert(std::has_unique_object_representations_v<int16_t>);
static_assert(std::has_unique_object_representations_v<int32_t>);
static_assert(std::has_unique_object_representations_v<int64_t>);
static_assert(std::has_unique_object_representations_v<void*>);

// C++ uses `std::nullptr_t` but doesn't make it have an object representation
// infuriatingly. Turn it back into a real pointer to fix this. This is
// obnoxiously annoying to do in C++ though for "reasons". For example, one
// might think to *just* use the type function below, but that won't work
// because we'll still at some point need to convert to a `const void*` value in
// generic code, and there is a *lot* of generic code below this dispatch level.
// One might then think, fine, *just* use the function that actually maps the
// value, but that's really annoying because it needs to pass along a `const &`
// on one arm, but it actually *can't* pass it along on the other arm because
// then it would return a `const &` to a local, because converting from a
// `nullptr` to a `const void*` amazingly **allocates a temporary**. Anyways,
// this was the least horrible assemblage of hacks I could come up with to
// locally make this odd case of C++ go away, but if there are better ways,
// please send patches.
template <typename T> struct MapNullPtrTToVoidPtrImpl {
  using Type = T;
};
template <> struct MapNullPtrTToVoidPtrImpl<std::nullptr_t> {
  using Type = const void*;
};
template <typename T>
using MapNullPtrTToVoidPtr = typename MapNullPtrTToVoidPtrImpl<T>::Type;

template <typename T>
inline auto MapNullPtrToVoidPtr(const T& value) -> const T& {
  // This overload should never be selected for `std::nullptr_t`, so
  // static_assert to get some better compiler error messages.
  static_assert(!std::is_same_v<T, std::nullptr_t>);
  return value;
}

// Overload to map `nullptr` values to actual pointer values that are null.
inline auto MapNullPtrToVoidPtr(std::nullptr_t /*value*/) -> const void* {
  return nullptr;
}

template <typename T>
constexpr bool NullPtrOrHasUniqueObjectRepresentations =
    std::has_unique_object_representations_v<MapNullPtrTToVoidPtr<T>>;

template <typename T, typename = std::enable_if_t<
                          NullPtrOrHasUniqueObjectRepresentations<T>>>
inline auto CarbonHash(HashState hash, const T& value) -> HashState {
  return HashState::Hash(std::move(hash), MapNullPtrToVoidPtr(value));
}

template <typename... Ts,
          typename = std::enable_if_t<
              (... && NullPtrOrHasUniqueObjectRepresentations<Ts>)>>
inline auto CarbonHash(HashState hash, const std::tuple<Ts...>& value)
    -> HashState {
  return std::apply(
      [&](const auto&... args) {
        return HashState::Hash(std::move(hash), MapNullPtrToVoidPtr(args)...);
      },
      value);
}

template <typename T, typename U,
          typename = std::enable_if_t<
              NullPtrOrHasUniqueObjectRepresentations<T> &&
              NullPtrOrHasUniqueObjectRepresentations<U> &&
              sizeof(T) <= sizeof(uint64_t) && sizeof(U) <= sizeof(uint64_t)>>
inline auto CarbonHash(HashState hash, const std::pair<T, U>& value)
    -> HashState {
  return CarbonHash(std::move(hash), std::tuple(value.first, value.second));
}

template <typename T, typename = std::enable_if_t<
                          std::has_unique_object_representations_v<T>>>
inline auto CarbonHash(HashState hash, llvm::ArrayRef<T> objs) -> HashState {
  return CarbonHash(
      std::move(hash),
      llvm::ArrayRef(reinterpret_cast<const std::byte*>(objs.data()),
                     objs.size() * sizeof(T)));
}

template <typename T>
inline auto CarbonHashDispatch(HashState hash, const T& value) -> HashState {
  // This unqualified call will find both the overloads in this namespace and
  // ADL-found functions in an associated namespace of `T`.
  return CarbonHash(std::move(hash), value);
}

}  // namespace Detail

template <typename T>
inline auto HashValue(const T& value, uint64_t seed) -> HashCode {
  return static_cast<HashCode>(
      Detail::CarbonHashDispatch(HashState(seed), value));
}

template <typename T>
inline auto HashValue(const T& value) -> HashCode {
  return HashValue(value, HashState::StaticRandomData[0]);
}

// # Implementation details below #

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
  hash.buffer = Mix(data0 ^ StaticRandomData[1],
                    data1 ^ StaticRandomData[3] ^ hash.buffer);
#else
  hash.buffer ^= Mix(data0 ^ StaticRandomData[1], data1 ^ StaticRandomData[3]);
#endif
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
    hash.buffer = Mix(data.first ^ StaticRandomData[(size - 1)],
                      data.second ^ hash.buffer);
#else
    hash.buffer =
        Mix(data.first ^ StaticRandomData[(size - 1) >> 1],
            data.second ^ StaticRandomData[(size - 1) & 0b11] ^ hash.buffer);
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
    hash.buffer ^= StaticRandomData[0];
    uint64_t m0 = Mix(Read8(data_ptr) ^ StaticRandomData[1],
                      Read8(data_ptr + 8) ^ hash.buffer);

    const std::byte* tail_16b_ptr = data_ptr + (size - 16);
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
  // For integer types up to 64-bit widths, we hash the 2's compliment value by
  // casting to unsigned and hashing as a `uint64_t`. This has the downside of
  // making negative integers hash differently based on their width, but
  // anything else would require potentially expensive sign extension. For
  // positive integers though, there is no problem with using differently sized
  // integers (for example literals) than the stored keys.
  if constexpr (sizeof(T) <= 8 &&
                (std::is_enum_v<T> || std::is_integral_v<T>)) {
    uint64_t ext_value = static_cast<std::make_unsigned_t<T>>(value);
    return HashOne(std::move(hash), ext_value);
  }

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

template <typename... Ts, typename /*enable_if*/>
inline auto HashState::Hash(HashState hash, const Ts&... value) -> HashState {
  if constexpr (sizeof...(Ts) == 0) {
    return HashOne(std::move(hash), 0);
  }
  if constexpr (sizeof...(Ts) == 1) {
    return Hash(std::move(hash), value...);
  }
  if constexpr ((... && (sizeof(Ts) <= 8))) {
    if constexpr (sizeof...(Ts) == 2) {
      return HashTwo(std::move(hash), ReadSmall(value)...);
    }

    // More than two, but all small -- read each one into a contiguous buffer of
    // data. This may be a bit memory wasteful by padding everything out to
    // 8-byte chunks, but for that regularity the hashing is likely faster.
    const uint64_t data[] = {ReadSmall(value)...};
    return Hash(std::move(hash), data);
  }

  // For larger objects, hash each one down to a hash code and then hash those
  // as a buffer.
  const uint64_t data[] = {
      static_cast<uint64_t>(static_cast<HashCode>(Hash(HashState(hash.buffer), value)))...};
  return Hash(std::move(hash), data);
}

}  // namespace Carbon

#endif  // CARBON_COMMON_HASHING_H_
