
// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_SMHASHER_STUB_H_
#define CARBON_COMMON_SMHASHER_STUB_H_

#include "absl/hash/hash.h"
#include "common/hashing.h"
#include "llvm/ADT/Hashing.h"

// This header provides inline functions that conform to the specific signature
// needed by the SMHasher framework. Those are the only routines that should be
// included here. They're then patched into the SMHasher global array of hash
// functions to test. We isolate everything inside a special sub-namespace as
// well.
namespace Carbon::SMHasher {

inline void HashTest(const void* key, int len, uint32_t seed, void* out) {
  HashState hash(seed);
  if (len <= 8) {
    uint64_t value;
    switch (len) {
      case 0:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 0>*>(key));
        break;
      case 1:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 1>*>(key));
        break;
      case 2:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 2>*>(key));
        break;
      case 3:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 3>*>(key));
        break;
      case 4:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 4>*>(key));
        break;
      case 5:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 5>*>(key));
        break;
      case 6:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 6>*>(key));
        break;
      case 7:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 7>*>(key));
        break;
      case 8:
        value = HashState::ReadSmall(
            *reinterpret_cast<const std::array<uint8_t, 8>*>(key));
        break;
    }
    hash = HashState::HashOne(std::move(hash), value);
  } else if (len <= 16) {
    auto values =
        HashState::Read8To16(reinterpret_cast<const std::byte*>(key), len);
    hash = HashState::HashTwo(std::move(hash), values.first, values.second);
  } else {
    // Hashing the size isn't relevant here, but is harmless, so fall back to a
    // common code path.
    hash = HashState::HashSizedBytes(
        std::move(hash), llvm::ArrayRef<std::byte>(
                             reinterpret_cast<const std::byte*>(key), len));
  }
  *static_cast<uint64_t*>(out) =
      static_cast<uint64_t>(static_cast<HashCode>(hash));
}

inline void SizedHashTest(const void* key, int len, uint32_t seed, void* out) {
  *static_cast<uint64_t*>(out) = static_cast<uint64_t>(
      HashValue(llvm::ArrayRef(reinterpret_cast<const std::byte*>(key), len),
                HashCode(seed)));
}

struct Bytes {
  const unsigned char* data;
  ssize_t size;
};

template <typename H>
inline auto AbslHashValue(H h, const Bytes& bytes) -> H {
  return H::combine_contiguous(std::move(h), bytes.data, bytes.size);
}

inline void LLVMHashTest(const void* key, int len, uint32_t /*seed*/,
                         void* out) {
  *static_cast<uint64_t*>(out) = llvm::hash_value(
      llvm::ArrayRef(reinterpret_cast<const std::byte*>(key), len));
}

inline void AbseilHashTest(const void* key, int len, uint32_t /*seed*/,
                           void* out) {
  Bytes bytes = {reinterpret_cast<const unsigned char*>(key), len};
  *static_cast<uint64_t*>(out) = absl::HashOf(bytes);
}

inline void AbseilSizedHashTest(const void* key, int len, uint32_t /*seed*/,
                                void* out) {
  std::string_view s(reinterpret_cast<const char*>(key), len);
  *static_cast<uint64_t*>(out) = absl::HashOf(s);
}

}  // namespace Carbon::SMHasher

#endif  // CARBON_COMMON_SMHASHER_STUB_H_
