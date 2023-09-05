// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashing.h"

#include <initializer_list>
#include <memory>

#include "llvm/Support/FormatVariadic.h"

namespace Carbon {

auto Detail::HashState::UpdateByteSequence(
    llvm::ArrayRef<std::byte> bytes) -> void {
  const std::byte* data = bytes.data();
  const ssize_t size = bytes.size();

  // First handle short sequences under 8 bytes.
  if (size < 8) {
    // Current AHash code has a special short-sequence lowering for strings.
    // This code uses that always. The general code path is preserved in a
    // `#if`-ed out block. The most notable difference is not pre-mixing the
    // size.
#if 1
    uint64_t data0;
    uint64_t data1;
    if (size >= 2) {
      if (size >= 4) {
        // 4-8 bytes use potentially overlapping 4 byte reads.
        data0 = Read32(data);
        data1 = Read32(data + size - 4);
      } else {
        // 2-3 bytes use overlapping 2 byte reads.
        data0 = Read16(data);
        data1 = Read16(data + size - 2);
      }
    } else {
      if (size > 0) {
        // Use the single byte twice.
        data0 = Read8(data);
        data1 = data0;
      } else {
        // TODO: This actually does *some* mixing -- is this needed?
        data0 = 0;
        data1 = 0;
      }
    }
    buffer = FoldedMultiply(data0 ^ buffer, data1 ^ extra_keys[1]);
    pad += size;
    return;
#else
    // Pre-mix the size into the buffer.
    //
    // Note that this avoids xor and uses addition to avoid carefully formed
    // input cancelling out updates.
    buffer += size;
    buffer *= MulConstant;

    if (size >= 2) {
      if (size >= 4) {
        // 4-8 bytes use potentially overlapping 4 byte reads.
        UpdateTwoChunks(Read32(data), Read32(data + size - 4));
        return;
      }

      // 2-3 bytes use overlapping 2 byte reads.
      UpdateTwoChunks(Read16(data), Read16(data + size - 2));
      return;
    }
    if (size > 0) {
      // Use the single byte twice.
      UpdateTwoChunks(Read8(data), Read8(data));
      return;
    }
    // TODO: This actually does *some* mixing -- is this needed?
    UpdateTwoChunks(0, 0);
    return;
#endif
  }

  // Pre-mix the size into the buffer.
  //
  // Note that this avoids xor and uses addition to avoid carefully formed
  // input cancelling out updates.
  buffer += size;
  buffer *= MulConstant;

  if (size <= 16) {
    // Use two overlapping 8-byte reads.
    UpdateTwoChunks(Read64(data), Read64(data + size - 8));
    return;
  }

  // Pre-mix the tail.
  // TODO: Evaluate if this is actually needed. Pre-mixing the tail is a
  // hard to guess at performance tradeoff. One one hand, it might hide
  // latency of a more likely misaligned read. On the other hand, it seems
  // much less likely to benefit from prefetching.
  uint64_t data0 = Read64(data + size - 16);
  uint64_t data1 = Read64(data + size - 8);
  UpdateTwoChunks(data0, data1);

  // Now mix the data in order except for the tail.
  const std::byte* end = data + size - 16;
  while (data < end) {
    UpdateTwoChunks(Read64(data), Read64(data + 8));
    data += 16;
  }
}

}  // namespace Carbon
