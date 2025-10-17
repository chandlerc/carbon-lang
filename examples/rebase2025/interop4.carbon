// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";

import Cpp inline '''

class Up {
 public:
  auto Count() const -> int {
    return count;
  }

  virtual auto Inc() -> void {
    count += 1;
  }

 protected:
  int count = 0;
};

''';

base class UpOrDown {
  extend base: Cpp.Up;

  override fn Inc[addr self: Self*]() {
    self.Add(1);
  }

  virtual fn Dec[addr self: Self*]() {
    self.Add(-1);
  }

  private fn Add[addr self: Self*](delta: i32) {
    self.count += delta;
  }
}

fn Run() -> i32 {
  var counter: UpOrDown;

  counter.Inc();
  Core.Print(counter.Count());
  counter.Dec();
  Core.Print(counter.Count());

  return 0;
}
