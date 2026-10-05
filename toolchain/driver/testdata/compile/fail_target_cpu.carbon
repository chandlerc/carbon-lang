// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ARGS: --include-diagnostic-kind compile --target=x86_64-unknown-linux-gnu --target-cpu=not-a-valid-cpu foo.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/driver/testdata/compile/fail_target_cpu.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/driver/testdata/compile/fail_target_cpu.carbon
// CHECK:STDERR: error: unknown target CPU 'not-a-valid-cpu' [CppInteropDriverError]
// CHECK:STDERR:

// --- foo.carbon
