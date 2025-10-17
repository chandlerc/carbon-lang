// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";

import Cpp inline #'''c++

auto Double(int value) -> int { return value * 2; }
auto Double(float value) -> float { return value * 2.0f; }

auto Round(float value) -> int { return static_cast<int>(value); }

'''#;

fn Run() -> i32 {
  let x: i32 = Cpp.Double(21);
  let y: f32 = Cpp.Double(0.5 as f32);
  Core.Print(x);
  Core.Print(Cpp.Round(y));

  return 0;
}
