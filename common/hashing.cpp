// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashing.h"

#include <array>
#include <initializer_list>
#include <memory>

#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

#if 1
// Random data taken from the hexadecimal digits of Pi's fractional component,
// written in lexical order for convenience of reading. The resulting
// byte-stream will be different due to little-endian integers. The initializers
// here can be generated with the following shell script:
//
// ```sh
// echo 'obase=16; scale=308; 4*a(1)' | env BC_LINE_LENGTH=500 bc -l \
//  | cut -c 3- | tr '[:upper:]' '[:lower:]' \
//  | sed -e "s/.\{4\}/&'/g" \
//  | sed -e "s/\(.\{4\}'.\{4\}'.\{4\}'.\{4\}\)'/0x\1,\n/g"
// ```
constexpr std::array<uint64_t, 16> HashState::StaticRandomData = {
    0x243f'6a88'85a3'08d3, 0x1319'8a2e'0370'7344, 0xa409'3822'299f'31d0,
    0x082e'fa98'ec4e'6c89, 0x4528'21e6'38d0'1377, 0xbe54'66cf'34e9'0c6c,
    0xc0ac'29b7'c97c'50dd, 0x3f84'd5b5'b547'0917, 0x9216'd5d9'8979'fb1b,
    0xd131'0ba6'98df'b5ac, 0x2ffd'72db'd01a'dfb7, 0xb8e1'afed'6a26'7e96,
    0xba7c'9045'f12c'7f99, 0x24a1'9947'b391'6cf7, 0x0801'f2e2'858e'fc16,
    0x6369'20d8'7157'4e68,
};
#else
constexpr std::array<uint64_t, 8> HashState::StaticRandomData = {
    0xa2cc'5728'5aa3'6f15, 0xac34'2eed'8454'fc11, 0x8c09'ddc3'5ac4'a3eb,
    0xcc61'97d7'3e83'dddf, 0xc68f'1314'293f'5b77, 0xadd3'daca'21f8'8fb5,
    0x979a'170c'93b4'd209, 0x8446'a70c'9065'1a0f,
};
#endif

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
  auto global_address =
      reinterpret_cast<uintptr_t>(&HashState::global_variable);
  volatile char local_variable;
  auto local_address = reinterpret_cast<uintptr_t>(&local_variable);

  // Make sure the memory for these variables lives and is distinct.
  HashState::global_variable = 1;
  local_variable = 2;

  // Compute hashes to provide a rough RNG seeded by the ASLR entropy above.
  // This code manually mixes the initial state buffer with statically computed
  // data. We can then use the `HashOne` routine that doesn't depend on this
  // random data.
  HashState seed_hash;
  seed_hash.buffer = Mix(global_address ^ StaticRandomData[0],
                         local_address ^ StaticRandomData[2]);
  // Each round of hashing past this should mix the bits more completely, and
  // the most important one is at index 0 as we use that as the initial seed so
  // initialize the data in reverse order.
  for (int i : llvm::reverse(llvm::seq(data.size()))) {
    seed_hash = RotState(std::move(seed_hash));
    seed_hash = HashOne(
        std::move(seed_hash),
        ((i & 1) ? global_address : local_address) ^ StaticRandomData[i]);
    data[i] = static_cast<uint64_t>(static_cast<HashCode>(seed_hash));
  }

  return data;
}

auto HashState::DumpRandomData() -> void {
  llvm::errs() << "Random hashing state for this process:\n";
  for (auto x : RandomData) {
    llvm::errs() << "  " << llvm::formatv("{0:xd}", x) << "\n";
  }
}

auto HashState::HashSizedBytesLarge(HashState hash,
                                    llvm::ArrayRef<std::byte> bytes)
    -> HashState {
  const std::byte* data_ptr = bytes.data();
  const ssize_t size = bytes.size();
  CARBON_DCHECK(size > 32);

  __builtin_prefetch(data_ptr, 0, 0);

  // If we have more than 32 bytes, we're going to handle two 32-byte chunks
  // at a time using a simplified version of the main algorithm. This is based
  // heavily on the 64-byte and larger processing approach used by Abseil. The
  // goal is to mix the input data using as few multiplies (or other
  // operations) as we can and with as much ILP as we can. The ILP comes
  // largely from creating parallel structures to the operations.
  auto mix32 = [](const std::byte* data_ptr, uint64_t buffer, uint64_t random0,
                  uint64_t random1) {
    uint64_t a = Read8(data_ptr);
    uint64_t b = Read8(data_ptr + 8);
    uint64_t c = Read8(data_ptr + 16);
    uint64_t d = Read8(data_ptr + 24);
    uint64_t m0 = Mix(a ^ random0, b ^ buffer);
    uint64_t m1 = Mix(c ^ random1, d ^ buffer);
    return (m0 ^ m1);
  };

  uint64_t buffer0 = hash.buffer ^ StaticRandomData[0];
  uint64_t buffer1 = hash.buffer ^ StaticRandomData[2];
  const std::byte* tail_32b_ptr = data_ptr + (size - 32);
  const std::byte* end_ptr = data_ptr + (size - 64);
  while (data_ptr < end_ptr) {
    // Prefetch the next cacheline.
    __builtin_prefetch(data_ptr + 64, 0, 0);

    buffer0 =
        mix32(data_ptr, buffer0, StaticRandomData[4], StaticRandomData[5]);
    buffer1 =
        mix32(data_ptr + 32, buffer1, StaticRandomData[6], StaticRandomData[7]);

    data_ptr += 64;
  }

  if (data_ptr < tail_32b_ptr) {
    buffer0 =
        mix32(data_ptr, buffer0, StaticRandomData[4], StaticRandomData[5]);
  }
  buffer1 =
      mix32(tail_32b_ptr, buffer1, StaticRandomData[6], StaticRandomData[7]);

  hash.buffer = buffer0 ^ buffer1;
  hash = RotState(std::move(hash));
  hash = HashOne(std::move(hash), size);
  return hash;
}

}  // namespace Carbon
