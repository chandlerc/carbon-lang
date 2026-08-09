// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// TODO: Add ranges and switch to "--dump-sem-ir-ranges=only".
// EXTRA-ARGS: --dump-sem-ir-ranges=if-present
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/function/definition/fail_decl_param_mismatch.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/function/definition/fail_decl_param_mismatch.carbon

fn F();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+4]]:1-20: error: redeclaration differs because of parameter count of 1 [RedeclParamCountDiffers]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:13-17: note: this parameter has no counterpart [RedeclParamWithoutCounterpart]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-3]]:1-7: note: previously declared with parameter count of 0 [RedeclParamCountPrevious]
// CHECK:STDERR:
fn F(unused x: ()) {}

fn G(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+4]]:1-8: error: redeclaration differs because of parameter count of 0 [RedeclParamCountDiffers]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:6-10: note: this parameter has no counterpart [RedeclParamWithoutCounterpart]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-3]]:1-12: note: previously declared with parameter count of 1 [RedeclParamCountPrevious]
// CHECK:STDERR:
fn G() {}

fn H(x: ());
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+2]]:16-19: error: `Core.Bool` implicitly referenced here, but package `Core` not found [CoreNotFound]
// CHECK:STDERR:
fn H(unused x: bool) {}

fn I();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:11-12: error: function redeclaration differs because return type is `()` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:1-7: note: previously declared with no return type [FunctionRedeclReturnTypePreviousNoReturn]
// CHECK:STDERR:
fn I() -> () { return (); }

fn J() -> ();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:1-8: error: function redeclaration differs because no return type is provided [FunctionRedeclReturnTypeDiffersNoReturn]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:11-12: note: previously declared with return type `()` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR:
fn J() {}

fn K() -> ();
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE+3]]:11-12: error: function redeclaration differs because return type is `{}` [FunctionRedeclReturnTypeDiffers]
// CHECK:STDERR: fail_decl_param_mismatch.carbon:[[@LINE-2]]:11-12: note: previously declared with return type `()` [FunctionRedeclReturnTypePrevious]
// CHECK:STDERR:
fn K() -> {} { return {}; }

// CHECK:STDOUT: --- fail_decl_param_mismatch.carbon
// CHECK:STDOUT:
// CHECK:STDOUT: constants {
// CHECK:STDOUT:   %F.type.eb3ec9.1: type = fn_type @F.loc15 [concrete]
// CHECK:STDOUT:   %empty_tuple.type: type = tuple_type () [concrete]
// CHECK:STDOUT:   %F.7088cc.1: %F.type.eb3ec9.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_tuple: %empty_tuple.type = tuple_value () [concrete]
// CHECK:STDOUT:   %pattern_type.cb1: type = pattern_type %empty_tuple.type [concrete]
// CHECK:STDOUT:   %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete]
// CHECK:STDOUT:   %x.patt.2180a6.1: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %F.type.eb3ec9.2: type = fn_type @F.loc20 [concrete]
// CHECK:STDOUT:   %F.7088cc.2: %F.type.eb3ec9.2 = struct_value () [concrete]
// CHECK:STDOUT:   %x.patt.2180a6.2: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete]
// CHECK:STDOUT:   %G.type.f42045.1: type = fn_type @G.loc22 [concrete]
// CHECK:STDOUT:   %G.4228ed.1: %G.type.f42045.1 = struct_value () [concrete]
// CHECK:STDOUT:   %G.type.f42045.2: type = fn_type @G.loc27 [concrete]
// CHECK:STDOUT:   %G.4228ed.2: %G.type.f42045.2 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.c0bf33.1: type = fn_type @H.loc29 [concrete]
// CHECK:STDOUT:   %H.aacb46.1: %H.type.c0bf33.1 = struct_value () [concrete]
// CHECK:STDOUT:   %H.type.c0bf33.2: type = fn_type @H.loc32 [concrete]
// CHECK:STDOUT:   %H.aacb46.2: %H.type.c0bf33.2 = struct_value () [concrete]
// CHECK:STDOUT:   %I.type.4c8936.1: type = fn_type @I.loc34 [concrete]
// CHECK:STDOUT:   %I.8a67f5.1: %I.type.4c8936.1 = struct_value () [concrete]
// CHECK:STDOUT:   %.262: Core.Form = init_form %empty_tuple.type [concrete]
// CHECK:STDOUT:   %return.param_patt.900: %pattern_type.cb1 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.427: %pattern_type.cb1 = return_slot_pattern %return.param_patt.900, %empty_tuple.type [concrete]
// CHECK:STDOUT:   %I.type.4c8936.2: type = fn_type @I.loc38 [concrete]
// CHECK:STDOUT:   %I.8a67f5.2: %I.type.4c8936.2 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.6304f1.1: type = fn_type @J.loc40 [concrete]
// CHECK:STDOUT:   %J.64b370.1: %J.type.6304f1.1 = struct_value () [concrete]
// CHECK:STDOUT:   %J.type.6304f1.2: type = fn_type @J.loc44 [concrete]
// CHECK:STDOUT:   %J.64b370.2: %J.type.6304f1.2 = struct_value () [concrete]
// CHECK:STDOUT:   %K.type.b73f4f.1: type = fn_type @K.loc46 [concrete]
// CHECK:STDOUT:   %K.4fd1e7.1: %K.type.b73f4f.1 = struct_value () [concrete]
// CHECK:STDOUT:   %empty_struct_type: type = struct_type {} [concrete]
// CHECK:STDOUT:   %empty_struct: %empty_struct_type = struct_value () [concrete]
// CHECK:STDOUT:   %.469: Core.Form = init_form %empty_struct_type [concrete]
// CHECK:STDOUT:   %pattern_type.a96: type = pattern_type %empty_struct_type [concrete]
// CHECK:STDOUT:   %return.param_patt.296: %pattern_type.a96 = out_param_pattern [concrete]
// CHECK:STDOUT:   %return.patt.c45: %pattern_type.a96 = return_slot_pattern %return.param_patt.296, %empty_struct_type [concrete]
// CHECK:STDOUT:   %K.type.b73f4f.2: type = fn_type @K.loc50 [concrete]
// CHECK:STDOUT:   %K.4fd1e7.2: %K.type.b73f4f.2 = struct_value () [concrete]
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: file {
// CHECK:STDOUT:   package: <namespace> = namespace [concrete] {
// CHECK:STDOUT:     .F = %F.decl.loc15
// CHECK:STDOUT:     .G = %G.decl.loc22
// CHECK:STDOUT:     .H = %H.decl.loc29
// CHECK:STDOUT:     .I = %I.decl.loc34
// CHECK:STDOUT:     .J = %J.decl.loc40
// CHECK:STDOUT:     .K = %K.decl.loc46
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %F.decl.loc15: %F.type.eb3ec9.1 = fn_decl @F.loc15 [concrete = constants.%F.7088cc.1] {} {}
// CHECK:STDOUT:   %F.decl.loc20: %F.type.eb3ec9.2 = fn_decl @F.loc20 [concrete = constants.%F.7088cc.2] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.2180a6.1]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc20_17.1: type = splice_block %.loc20_17.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc20_17.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc20_17.3: type = converted %.loc20_17.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc22: %G.type.f42045.1 = fn_decl @G.loc22 [concrete = constants.%G.4228ed.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.2180a6.2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc22_10.1: type = splice_block %.loc22_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc22_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc22_10.3: type = converted %.loc22_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %G.decl.loc27: %G.type.f42045.2 = fn_decl @G.loc27 [concrete = constants.%G.4228ed.2] {} {}
// CHECK:STDOUT:   %H.decl.loc29: %H.type.c0bf33.1 = fn_decl @H.loc29 [concrete = constants.%H.aacb46.1] {
// CHECK:STDOUT:     %x.param_patt: %pattern_type.cb1 = value_param_pattern [concrete = constants.%x.param_patt]
// CHECK:STDOUT:     %x.patt: %pattern_type.cb1 = wrapper_binding_pattern x, %x.param_patt [concrete = constants.%x.patt.2180a6.2]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: %empty_tuple.type = value_param call_param0
// CHECK:STDOUT:     %.loc29_10.1: type = splice_block %.loc29_10.3 [concrete = constants.%empty_tuple.type] {
// CHECK:STDOUT:       %.loc29_10.2: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:       %.loc29_10.3: type = converted %.loc29_10.2, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     }
// CHECK:STDOUT:     %x: %empty_tuple.type = wrapper_binding x, %x.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %H.decl.loc32: %H.type.c0bf33.2 = fn_decl @H.loc32 [concrete = constants.%H.aacb46.2] {
// CHECK:STDOUT:     %x.param_patt: <error> = value_param_pattern [concrete = <error>]
// CHECK:STDOUT:     %x.patt: <error> = wrapper_binding_pattern x, %x.param_patt [concrete = <error>]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %x.param: <error> = value_param call_param0
// CHECK:STDOUT:     %.loc32: type = type_literal <error> [concrete = <error>]
// CHECK:STDOUT:     %x: <error> = wrapper_binding x, <error> [concrete = <error>]
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %I.decl.loc34: %I.type.4c8936.1 = fn_decl @I.loc34 [concrete = constants.%I.8a67f5.1] {} {}
// CHECK:STDOUT:   %I.decl.loc38: %I.type.4c8936.2 = fn_decl @I.loc38 [concrete = constants.%I.8a67f5.2] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc38_12.2 [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc38_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc38_12.2: type = converted %.loc38_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc38_12.3: Core.Form = init_form %.loc38_12.2 [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl.loc40: %J.type.6304f1.1 = fn_decl @J.loc40 [concrete = constants.%J.64b370.1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc40_12.2 [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc40_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc40_12.2: type = converted %.loc40_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc40_12.3: Core.Form = init_form %.loc40_12.2 [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %J.decl.loc44: %J.type.6304f1.2 = fn_decl @J.loc44 [concrete = constants.%J.64b370.2] {} {}
// CHECK:STDOUT:   %K.decl.loc46: %K.type.b73f4f.1 = fn_decl @K.loc46 [concrete = constants.%K.4fd1e7.1] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.cb1 = out_param_pattern [concrete = constants.%return.param_patt.900]
// CHECK:STDOUT:     %return.patt: %pattern_type.cb1 = return_slot_pattern %return.param_patt, %.loc46_12.2 [concrete = constants.%return.patt.427]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc46_12.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:     %.loc46_12.2: type = converted %.loc46_12.1, constants.%empty_tuple.type [concrete = constants.%empty_tuple.type]
// CHECK:STDOUT:     %.loc46_12.3: Core.Form = init_form %.loc46_12.2 [concrete = constants.%.262]
// CHECK:STDOUT:     %return.param: ref %empty_tuple.type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_tuple.type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT:   %K.decl.loc50: %K.type.b73f4f.2 = fn_decl @K.loc50 [concrete = constants.%K.4fd1e7.2] {
// CHECK:STDOUT:     %return.param_patt: %pattern_type.a96 = out_param_pattern [concrete = constants.%return.param_patt.296]
// CHECK:STDOUT:     %return.patt: %pattern_type.a96 = return_slot_pattern %return.param_patt, %.loc50_12.2 [concrete = constants.%return.patt.c45]
// CHECK:STDOUT:   } {
// CHECK:STDOUT:     %.loc50_12.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:     %.loc50_12.2: type = converted %.loc50_12.1, constants.%empty_struct_type [concrete = constants.%empty_struct_type]
// CHECK:STDOUT:     %.loc50_12.3: Core.Form = init_form %.loc50_12.2 [concrete = constants.%.469]
// CHECK:STDOUT:     %return.param: ref %empty_struct_type = out_param call_param0
// CHECK:STDOUT:     %return: ref %empty_struct_type = return_slot %return.param
// CHECK:STDOUT:   }
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc15();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @F.loc20(%x.param: %empty_tuple.type) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.loc22(%x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @G.loc27() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.loc29(%x.param: %empty_tuple.type);
// CHECK:STDOUT:
// CHECK:STDOUT: fn @H.loc32(%x.param: <error>) {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.loc34();
// CHECK:STDOUT:
// CHECK:STDOUT: fn @I.loc38() -> out %return.param: %empty_tuple.type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc38_24.1: %empty_tuple.type = tuple_literal () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38_24.2: init %empty_tuple.type = tuple_init () [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   %.loc38_25: init %empty_tuple.type = converted %.loc38_24.1, %.loc38_24.2 [concrete = constants.%empty_tuple]
// CHECK:STDOUT:   return %.loc38_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J.loc40() -> out %return.param: %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @J.loc44() {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   return
// CHECK:STDOUT: }
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.loc46() -> out %return.param: %empty_tuple.type;
// CHECK:STDOUT:
// CHECK:STDOUT: fn @K.loc50() -> out %return.param: %empty_struct_type {
// CHECK:STDOUT: !entry:
// CHECK:STDOUT:   %.loc50_24.1: %empty_struct_type = struct_literal () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc50_24.2: init %empty_struct_type = struct_init () [concrete = constants.%empty_struct]
// CHECK:STDOUT:   %.loc50_25: init %empty_struct_type = converted %.loc50_24.1, %.loc50_24.2 [concrete = constants.%empty_struct]
// CHECK:STDOUT:   return %.loc50_25
// CHECK:STDOUT: }
// CHECK:STDOUT:
