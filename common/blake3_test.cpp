// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/blake3.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

using ::testing::Eq;

namespace Carbon::Testing {
namespace {

static_assert(sizeof(Blake3Hash) == 32, "Hash size doesn't match Blake3 spec.");

constexpr struct {
  int size;
  Blake3Hash hash;
} TestVectors[] = {
    {0,
     {
         0xaf'13'49'b9u,
         0xf5'f9'a1'a6u,
         0xa0'40'4d'eau,
         0x36'dc'c9'49u,
         0x9b'cb'25'c9u,
         0xad'c1'12'b7u,
         0xcc'9a'93'cau,
         0xe4'1f'32'62u,
     }},
};

TEST(Blake3Hash, TestVectors) {
  EXPECT_THAT(Blake3Hash::Hash(""), Eq(TestVectors[0].hash));
}

}  // namespace
}  // namespace Carbon::Testing
