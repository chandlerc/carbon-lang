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

  explicit HashCode(size_t value) : value_(value) {}

  explicit operator size_t() const { return value_; }

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
  size_t value_ = 0;
};

namespace Detail {

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
constexpr uint64_t RandomData[8] = {
    0x243f'6a88'85a3'08d3, 0x1319'8a2e'0370'7344, 0xa409'3822'299f'31d0,
    0x082e'fa98'ec4e'6c89, 0x4528'21e6'38d0'1377, 0xbe54'66cf'34e9'0c6c,
    0xc0ac'29b7'c97c'50dd, 0x3f84'd5b5'b547'0913,
};

// This constant from Knuth's PRNG. Claimed to work better than others from
// "splitmix32" by the AHash authors.
constexpr uint64_t MulConstant = 6364136223846793005;

// Undocumented constant from AHash for rotations.
constexpr uint64_t RotConstant = 23;

inline auto FoldedMultiply(uint64_t lhs, uint64_t rhs) -> uint64_t {
  // Use the C23 extended integer support that Clang provides as a general
  // language extension.
  using U128 = unsigned _BitInt(128);
  U128 result = static_cast<U128>(lhs) * static_cast<U128>(rhs);
  return static_cast<uint64_t>(result) ^ static_cast<uint64_t>(result >> 64);
}

inline auto Read8(const std::byte *data) -> uint64_t {
  uint8_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto Read16(const std::byte *data) -> uint64_t {
  uint16_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto Read32(const std::byte *data) -> uint64_t {
  uint32_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

inline auto Read64(const std::byte *data) -> uint64_t {
  uint64_t result;
  std::memcpy(&result, data, sizeof(result));
  return result;
}

struct HashState {
  static auto Create() -> HashState { return {}; }

  inline auto Update(uint64_t data) -> void {
    buffer = FoldedMultiply(data ^ buffer, MulConstant);
  }

  inline auto UpdateTwoChunks(uint64_t data0, uint64_t data1) -> void  {
    uint64_t combined = FoldedMultiply(data0 ^ extra_keys[0], data1 ^ extra_keys[1]);
    buffer = llvm::rotl((buffer + pad) ^ combined, RotConstant);
  }
  
  auto UpdateByteSequence(llvm::ArrayRef<std::byte> bytes) -> void;

  inline auto ShortFinish() -> HashCode {
    return HashCode(buffer + pad);
  }

  inline auto Finish() -> HashCode {
    return HashCode(llvm::rotl(FoldedMultiply(buffer, pad), buffer & 63));
  }

  uint64_t buffer = RandomData[0];
  uint64_t pad = RandomData[1];
  uint64_t extra_keys[2] = {RandomData[2], RandomData[3]};
};

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

// FIXME: This should use an extensible API.
inline auto HashValue(llvm::StringRef s) -> HashCode {
  auto state = Detail::HashState::Create();
  llvm::ArrayRef<std::byte> bytes(reinterpret_cast<const std::byte*>(s.data()),
                                  s.size());
  state.UpdateByteSequence(bytes);
  return state.Finish();
}

}  // namespace Carbon

#endif  // CARBON_COMMON_HASHING_H_
