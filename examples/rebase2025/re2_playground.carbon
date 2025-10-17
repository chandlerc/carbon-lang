// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Cpp library "re2/re2.h";

import Cpp inline #'''c++
auto Log(std::string_view sv) -> void {
  printf("%.*s\n", (int)sv.size(), sv.data());
}
'''#;

class Regexp {
  extend base: Cpp.re2.RE2;

  fn Make(re: str) -> Self {
    return {.base = RE2(re)};
  }
  fn Submatch[addr self: Self*](text: str) -> str {
    var submatch: str;
    if (FullMatch(text, *self as Cpp.re2.RE2, &submatch)) {
      return submatch;
    }
    return "Match failed!";
  }
}

fn Run() -> i32 {
  var re: Regexp = Regexp.Make(".* (.*)!");
  Cpp.Log(re.Submatch("Hello, Rebase!"));
  return 0;
}
