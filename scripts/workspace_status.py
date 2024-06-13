#!/usr/bin/env python3

"""Bazel `--workspace_status_command` script.

This script is designed to be used in Bazel`s `--workspace_status_command` and
generate any desirable status artifacts.
"""

__copyright__ = """
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""

import subprocess


def git_hash() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], shell=False, text=True
    ).strip()


def git_is_dirty() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"], shell=False, text=True
    ).strip()


def main() -> None:
    print("STABLE_TEST_KEY " + git_hash())


if __name__ == "__main__":
    main()
