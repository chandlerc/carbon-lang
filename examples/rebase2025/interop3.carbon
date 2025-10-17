// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import Core library "io";

import Cpp inline #'''c++

class Point {
 public:
  Point(int x, int y) : x_(x), y_(y) {}

  auto GetX() const -> int { return x_; }
  auto GetY() const -> int { return y_; }

  auto SetX(int new_x) -> void { x_ = new_x; }
  auto SetY(int new_y) -> void { y_ = new_y; }

 private:
  int x_;
  int y_;
};

'''#;

fn Run() -> i32 {
  var p: Cpp.Point = Cpp.Point.Point(1, 2);
  Core.Print(p.GetX());
  Core.Print(p.GetY());

  p.SetX(3);
  Core.Print(p.GetX());
  Core.Print(p.GetY());

  return 0;
}
