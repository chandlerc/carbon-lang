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

const std::array<uint64_t, 8> HashState::RandomData =
    HashState::ComputeRandomData();

volatile char HashState::global_variable;

auto HashState::ComputeRandomData() -> std::array<uint64_t, 8> {
  std::array<uint64_t, 8> data;

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
  // Each round of hashing past this should mix the bits more completely, and
  // the most important one is the first entry that we use as the initial seed
  // so initialize the data in reverse order.
  data[7] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), global_address);
  data[6] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), local_address);
  data[5] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), global_address);
  data[4] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), local_address);
  data[3] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), global_address);
  data[2] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), local_address);
  data[1] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  seed_hash = HashOne(std::move(seed_hash), global_address);
  data[0] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));

  return data;
}

auto HashState::HashSizedBytesLarge(HashState hash, llvm::ArrayRef<std::byte> bytes)
    -> HashState {
  const std::byte* data_ptr = bytes.data();
  const ssize_t size = bytes.size();
  CARBON_DCHECK(size > 16);

  __builtin_prefetch(data_ptr, 0, 0);
#if 1
  const std::byte* tail_ptr = data_ptr + (size - 16);

  if (size > 64) {
    // If we have more than 64 bytes, we're going to handle chunks of 64 bytes
    // at a time using a simplified version of the main algorithm. This is based
    // heavily on the corresponding 64-byte processing approach used by Abseil.
    // The goal is to mix the 64-bytes of input data using as few multiplies (or
    // other operations) as we can and with as much ILP as we can. The ILP comes
    // largely from creating parallel structures to the operations.
    uint64_t buffer0 = hash.buffer;
    uint64_t buffer1 = hash.buffer;
    const std::byte* end_ptr = data_ptr + (size - 64);
    do {
      // Always prefetch the next cacheline.
      __builtin_prefetch(data_ptr + 64, 0, 0);
      //PrefetchToLocalCache(ptr + ABSL_CACHELINE_SIZE);

      uint64_t a = Read8(data_ptr);
      uint64_t b = Read8(data_ptr + 8);
      uint64_t c = Read8(data_ptr + 16);
      uint64_t d = Read8(data_ptr + 24);
      uint64_t cs0 = Mix(a ^ RandomData[4], b ^ buffer0);
      uint64_t cs1 = Mix(c ^ RandomData[5], d ^ buffer0);
      buffer0 = (cs0 ^ cs1);

      uint64_t e = Read8(data_ptr + 32);
      uint64_t f = Read8(data_ptr + 40);
      uint64_t g = Read8(data_ptr + 48);
      uint64_t h = Read8(data_ptr + 56);
      uint64_t ds0 = Mix(e ^ RandomData[6], f ^ buffer1);
      uint64_t ds1 = Mix(g ^ RandomData[7], h ^ buffer1);
      buffer1 = (ds0 ^ ds1);

      data_ptr += 64;
    } while (data_ptr < end_ptr);

    hash.buffer = buffer0 ^ buffer1;
    hash = MixState(std::move(hash));
  }

  while (data_ptr < tail_ptr) {
    hash = HashTwo(std::move(hash), Read8(data_ptr), Read8(data_ptr + 8));
    hash = MixState(std::move(hash));
    data_ptr += 16;
  }
  hash = HashTwo(std::move(hash), Read8(tail_ptr), Read8(tail_ptr + 8));
  hash = MixState(std::move(hash));
  hash = HashOne(std::move(hash), size);
#else
  hash.buffer ^= RandomData[0];

  if (bytes.size() > 64) {
    // If we have more than 64 bytes, we're going to handle chunks of 64
    // bytes at a time. We're going to build up two separate hash states
    // which we will then hash together.
    uint64_t duplicated_state = hash.buffer;

    do {
      // Always prefetch the next cacheline.
      __builtin_prefetch(bytes.data() + 64, 0, 0);
      //PrefetchToLocalCache(ptr + ABSL_CACHELINE_SIZE);

      uint64_t a = Read8(bytes.data());
      uint64_t b = Read8(bytes.data() + 8);
      uint64_t c = Read8(bytes.data() + 16);
      uint64_t d = Read8(bytes.data() + 24);
      uint64_t e = Read8(bytes.data() + 32);
      uint64_t f = Read8(bytes.data() + 40);
      uint64_t g = Read8(bytes.data() + 48);
      uint64_t h = Read8(bytes.data() + 56);

      uint64_t cs0 = Mix(a ^ RandomData[1], b ^ hash.buffer);
      uint64_t cs1 = Mix(c ^ RandomData[2], d ^ hash.buffer);
      hash.buffer = (cs0 ^ cs1);

      uint64_t ds0 = Mix(e ^ RandomData[3], f ^ duplicated_state);
      uint64_t ds1 = Mix(g ^ RandomData[4], h ^ duplicated_state);
      duplicated_state = (ds0 ^ ds1);

      bytes = bytes.drop_front(64);
    } while (bytes.size() > 64);

    hash.buffer ^= duplicated_state;
  }

  // We now have a data `ptr` with at most 64 bytes and the current state
  // of the hashing state machine stored in current_state.
  while (bytes.size() > 16) {
    uint64_t a = Read8(bytes.data());
    uint64_t b = Read8(bytes.data() + 8);

    hash.buffer = Mix(a ^ RandomData[1], b ^ hash.buffer);
    bytes = bytes.drop_front(16);
  }

  // We now have a data `ptr` with at most 16 bytes.
  uint64_t a = 0;
  uint64_t b = 0;
  if (bytes.size() > 8) {
    std::tie(a, b) = Read8To16(bytes.data(), bytes.size());
  } else if (bytes.size() > 3) {
    a = Read4To8(bytes.data(), bytes.size());
  } else if (bytes.size() > 0) {
    a = Read1To3(bytes.data(), bytes.size());
    b = 0;
  } else {
    a = 0;
    b = 0;
  }

  uint64_t w = Mix(a ^ RandomData[5], b ^ hash.buffer);
  uint64_t z = RandomData[6] ^ size;
  hash.buffer = Mix(w, z);
#endif
  return hash;
}

}  // namespace Carbon
