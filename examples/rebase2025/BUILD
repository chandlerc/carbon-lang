# Part of the Carbon Language project, under the Apache License v2.0 with LLVM
# Exceptions. See /LICENSE for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

load("//bazel/carbon_rules:defs.bzl", "carbon_binary")

carbon_binary(
    name = "hello_world",
    srcs = ["hello_world.carbon"],
)

carbon_binary(
    name = "hello_world_interop",
    srcs = ["hello_world_interop.carbon"],
)

carbon_binary(
    name = "interop1",
    srcs = ["interop1.carbon"],
)

carbon_binary(
    name = "interop2",
    srcs = ["interop2.carbon"],
)

carbon_binary(
    name = "interop3",
    srcs = ["interop3.carbon"],
)

carbon_binary(
    name = "interop4",
    srcs = ["interop4.carbon"],
    tags = ["manual"],
)

carbon_binary(
    name = "re2_playground",
    srcs = ["re2_playground.carbon"],
    deps = ["@re2"],
)
