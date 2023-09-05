// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/hashing.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "llvm/Support/FormatVariadic.h"
#include "testing/base/test_raw_ostream.h"

namespace Carbon::Testing {
namespace {

using ::testing::Eq;

TEST(HashingTest, BasicStateTests) {
  EXPECT_THAT(HashValue("a"), Eq(HashCode(0x969ba3b8def9bc10U)));
  EXPECT_THAT(HashValue("b"), Eq(HashCode(0xd76e45dd99624cbaU)));
  EXPECT_THAT(HashValue("aaaa"), Eq(HashCode(0x5340d1d57076dc83U)));
  EXPECT_THAT(HashValue("bbbbbbbb"), Eq(HashCode(0xc9b3a3c64a9b6fc7U)));
  EXPECT_THAT(HashValue("abcdefghijklmnopqrstuvwxyz"),
              Eq(HashCode(0x7ad7e2db2052d433U)));
  EXPECT_THAT(HashValue(""), Eq(HashCode(0x7d216c89612ace7fU)));
}

}  // namespace
}  // namespace Carbon::Testing
