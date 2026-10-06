// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <iostream>

#include "example_lib.h"

auto main() -> int {
  HelloWorld();
  std::cout.flush();
  int* volatile p = new int(42);
  delete p;
  return *p;
}
