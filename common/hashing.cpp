// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashing.h"

#include <array>
#include <initializer_list>
#include <memory>

#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

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
static constexpr std::array<uint64_t, 8> StaticRandomData = {
    0x243f'6a88'85a3'08d3, 0x1319'8a2e'0370'7344, 0xa409'3822'299f'31d0,
    0x082e'fa98'ec4e'6c89, 0x4528'21e6'38d0'1377, 0xbe54'66cf'34e9'0c6c,
    0xc0ac'29b7'c97c'50dd, 0x3f84'd5b5'b547'0913,
};

const std::array<uint64_t, 4> HashState::RandomData =
    HashState::ComputeRandomData();

volatile char HashState::global_variable;

auto HashState::ComputeRandomData() -> std::array<uint64_t, 4> {
  std::array<uint64_t, 4> data;

  // We want to provide entropy from program run to program run for use when
  // hashing. However, for debugging purposes it is useful to choose a source of
  // entropy that can be easily removed when debugging. The strategy we use is
  // to extract our entropy from ASLR so that when running under a debugger it
  // can be made reproducible. If this ceases to be a concern, we can gather
  // much more entropy by using `clock_gettime`, but this is only defending
  // against hash flooding and accidents, a much lower bar compared to
  // cryptographic uses of entropy.
  //
  // We use the address of a global variable with linkage, as well as the
  // address of a stack variable to find ASLR entropy.
  auto global_address = reinterpret_cast<uintptr_t>(&HashState::global_variable);
  volatile char local_variable;
  auto local_address = reinterpret_cast<uintptr_t>(&local_variable);

  // Make sure the memory for these variables lives and is distinct.
  HashState::global_variable = 1;
  local_variable = 2;

  HashState seed_hash;
  seed_hash.buffer = StaticRandomData[0];

  // Compute hashes to provide a rough RNG seeded by the ASLR above.
  seed_hash = HashTwoImpl(std::move(seed_hash), global_address,
                               local_address, StaticRandomData);
  data[0] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), global_address);
  data[1] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), local_address);
  data[2] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), global_address);
  data[3] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));

  return data;
}

inline auto Read1To3(const std::byte *data, ssize_t size) -> uint64_t {
  // Use carefully crafted indexing to avoid branches on the exact size while
  // reading.
  uint64_t byte0 = static_cast<uint8_t>(data[0]);
  uint64_t byte1 = static_cast<uint8_t>(data[size / 2]);
  uint64_t byte2 = static_cast<uint8_t>(data[size - 1]);
  return byte0 | (byte1 << ((size / 2) * 8)) | (byte2 << ((size - 1) * 8));
}

inline auto Read4To8(const std::byte *data, ssize_t size) -> uint64_t {
  uint32_t low;
  std::memcpy(&low, data, sizeof(low));
  uint32_t high;
  std::memcpy(&high, data + size - sizeof(high), sizeof(high));
  return low | (static_cast<uint64_t>(high) << ((size - sizeof(high)) * 8));
}

inline auto Read8To16(const std::byte* data, ssize_t size)
    -> std::pair<uint64_t, uint64_t> {
  uint64_t low;
  std::memcpy(&low, data, sizeof(low));
  uint64_t high;
  std::memcpy(&high, data + size - sizeof(high), sizeof(high));
  return {low, high};
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

auto HashState::HashSizedBytes(HashState hash, llvm::ArrayRef<std::byte> bytes)
    -> HashState {
  const std::byte* data = bytes.data();
  const ssize_t size = bytes.size();

  //UpdateSize(size);
  __builtin_prefetch(data, 0, 0);

  // First handle short sequences under 8 bytes.
  if (size <= 8) {
    uint64_t data0;
    uint64_t data1 = MulConstant;
    if (size == 8) {
      data0 = Read64(data);
    } else if (size == 4) {
      data0 = Read32(data);
    } else if (size > 4) {
      // 5-7 bytes use potentially overlapping 4 byte reads.
      data0 = Read32(data);
      data1 = Read32(data + size - 4) ^ RandomData[3];
    } else {
      if (size == 2) {
        data0 = Read16(data);
      } else if (size > 2) {
        // 3 bytes use overlapping 2 byte reads.
        data0 = Read16(data);
        data1 = Read16(data + size - 2) ^ RandomData[3];

      } else if (size > 0) {
        // Use the single byte twice.
        data0 = Read8(data);
        //data1 = data0;
      } else {
        // TODO: This actually does *some* mixing -- is this needed?
        data0 = 0;
        data1 = 0;
      }
    }
    hash.buffer = FoldedMultiply(data0 ^ hash.buffer, data1);
    //buffer = llvm::rotl(buffer + pad, RotConstant);

    return hash;
  }

  if (LLVM_LIKELY(size <= 16)) {
    // Use two overlapping 8-byte reads.
    return HashTwo(std::move(hash), Read64(data), Read64(data + size - 8));
    //UpdateSize(size);
  }

  const std::byte* tail = data + size - 16;
  if (LLVM_UNLIKELY(size > 32)) {
    const std::byte* end = data + size - 32;
    do {
      hash = HashTwo(std::move(hash), Read64(data), Read64(data + 8));
      hash = HashTwo(std::move(hash), Read64(data + 16), Read64(data + 24));
      data += 32;
    } while (data < end);
  }
  hash = HashTwo(std::move(hash), Read64(data), Read64(data + 8));
  hash = HashTwo(std::move(hash), Read64(tail), Read64(tail + 8));
  return hash;
}

}  // namespace Carbon
