// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_BLAKE3_H_
#define CARBON_COMMON_BLAKE3_H_

#include <array>

#include "llvm/ADT/StringRef.h"

namespace Carbon {

class Blake3Hash {
 public:
  std::array<uint32_t, 8> words;

  static auto Hash(llvm::StringRef data) -> Blake3Hash;

  auto operator==(const Blake3Hash& rhs) const -> bool {
    return words == rhs.words;
  }
  auto operator!=(const Blake3Hash& rhs) const -> bool {
    return words != rhs.words;
  }
};

}  // namespace Carbon

#endif  // CARBON_COMMON_BLAKE3_H_
