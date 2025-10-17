// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";

import Cpp inline '''

auto GetNumber() -> int {
  return 42;
}

''';

fn Run() -> i32 {
  let x: i32 = Cpp.GetNumber();
  Core.Print(x);

  return 0;
}
