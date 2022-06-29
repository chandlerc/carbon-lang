// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/blake3.h"

namespace Carbon {

static constexpr intptr_t BlockSize = 64;
static constexpr intptr_t ChunkSize = 1024;

static constexpr std::array<uint32_t, 8> IV = {
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
};

static constexpr std::array<intptr_t, 16> MessagePermutation = {
    2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8,
};

namespace {

struct ChunkState {
  std::array<uint32_t, 8> chaining_value;
  int64_t chunk_counter;
  std::array<unsigned char, BlockSize> block;
  int64_t block_length;
  int64_t blocks_compressed;
  uint32_t flags;
};

}  // namespace

auto Blake3Hash::Hash(llvm::StringRef /*data*/) -> Blake3Hash {
  return {0, 0, 0, 0, 0, 0, 0, 0};
}

}  // namespace Carbon
