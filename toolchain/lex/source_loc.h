// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_
#define CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_

#include "common/check.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::Lex {

// Where a lexer diagnostic points: a range of the source buffer, which is one
// character where it names a position rather than an extent.
//
// A position converts implicitly and a range has to be spelled out. That way
// round is what keeps a bare `const char*` from being taken for a range:
// `llvm::StringRef` builds itself from one by scanning to the next null, which
// in a source buffer is the end of the file.
class SourceLoc {
 public:
  // The position at `pos`, which must point into the source buffer.
  //
  // NOLINTNEXTLINE(google-explicit-constructor): A position is a location.
  SourceLoc(const char* pos) : text_(pos, 1) {}

  // The range `text` covers, which must point into the source buffer.
  explicit SourceLoc(llvm::StringRef text) : text_(text) {}

  // The range from `begin` up to `end`, both of which must point into the
  // source buffer.
  static auto Range(const char* begin, const char* end) -> SourceLoc {
    CARBON_CHECK(begin <= end, "A range's end can't come before its start.");
    return SourceLoc(llvm::StringRef(begin, end - begin));
  }

  // The source pointed at, starting at the position this names.
  auto text() const -> llvm::StringRef { return text_; }

 private:
  llvm::StringRef text_;
};

}  // namespace Carbon::Lex

#endif  // CARBON_TOOLCHAIN_LEX_SOURCE_LOC_H_
