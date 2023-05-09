#!/usr/bin/env python3

"""Runs clang-tidy over all Carbon files."""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import random
import string

NUM_CLASSES = 100000
NUM_METHODS = 45
NUM_FIELDS = 10


def random_id() -> str:
    length = random.randrange(3, 24)
    return "".join(random.choice(string.ascii_letters) for i in range(length))


def emit_field(i: int, k: int) -> None:
    name = random_id() + str(k)
    print(f"  var {name}: i32;")


def emit_method(i: int, j: int) -> None:
    name = random_id() + str(j)
    print(f"  // The {name} method declaration!")
    num_params = random.randrange(0, 4)
    params = ""
    for p in range(num_params):
        param_name = random_id() + str(p)
        params += f'{"" if p == 0 else ", "}{param_name}: i32'
    print(f"  fn {name}[self: Self]();")


def emit_class(i: int) -> None:
    name = random_id() + str(i)
    print(f"class {name} {{")

    for j in range(NUM_METHODS):
        emit_method(i, j)

    print()

    for k in range(NUM_FIELDS):
        emit_field(i, k)

    print("}")


for i in range(NUM_CLASSES):
    emit_class(i)
    print()
