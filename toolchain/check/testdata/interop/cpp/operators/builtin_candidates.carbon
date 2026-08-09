// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/full.carbon
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/check/testdata/interop/cpp/operators/builtin_candidates.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/check/testdata/interop/cpp/operators/builtin_candidates.carbon

// --- enum_operators.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
enum E { e1, e2, e3 };
enum class EC { e1, e2, e3 };
''';

fn TestEnum(a: Cpp.E, b: Cpp.E) {
  let _: Cpp.E = a & b;
  let _: Cpp.E = a | b;
  let _: Cpp.E = a ^ b;
  let _: Cpp.E = ^a;

  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

fn TestEnumClass(a: Cpp.EC, b: Cpp.EC) {
  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

// --- fail_enum_class_bitwise_operators.carbon

library "[[@TEST_NAME]]";

import Cpp inline '''c++
enum class EC { e1, e2, e3 };
''';

fn TestEnumClass(a: Cpp.EC, b: Cpp.EC) {
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+5]]:21: error: type `Cpp.EC` does not implement interface `Core.BitAndWith(Cpp.EC)` [MissingImplForType]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+4]]:21: note: `&` requires an impl of `Core.BitAndWith` [OperatorInterface]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+3]]:19: note: left operand has type `Cpp.EC` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.EC` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: Cpp.EC = a & b;
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+5]]:21: error: type `Cpp.EC` does not implement interface `Core.BitOrWith(Cpp.EC)` [MissingImplForType]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+4]]:21: note: `|` requires an impl of `Core.BitOrWith` [OperatorInterface]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+3]]:19: note: left operand has type `Cpp.EC` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.EC` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: Cpp.EC = a | b;
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+5]]:21: error: type `Cpp.EC` does not implement interface `Core.BitXorWith(Cpp.EC)` [MissingImplForType]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+4]]:21: note: `^` requires an impl of `Core.BitXorWith` [OperatorInterface]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+3]]:19: note: left operand has type `Cpp.EC` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.EC` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: Cpp.EC = a ^ b;
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+4]]:19: error: type `Cpp.EC` does not implement interface `Core.BitComplement` [MissingImplForType]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+3]]:19: note: `^` requires an impl of `Core.BitComplement` [OperatorInterface]
  // CHECK:STDERR: fail_enum_class_bitwise_operators.carbon:[[@LINE+2]]:20: note: operand has type `Cpp.EC` [UnaryOperatorOperandType]
  // CHECK:STDERR:
  let _: Cpp.EC = ^a;
}

// --- enum_operators_with_conversion.carbon

import Cpp inline '''c++
enum E {};
struct ConvertibleToEnum {
  operator E() const;
};
''';

fn TestEnumVsConvertibleToEnum(a: Cpp.E, b: Cpp.ConvertibleToEnum) {
  let _: Cpp.E = a & b;
  let _: Cpp.E = a | b;
  let _: Cpp.E = a ^ b;
  let _: Cpp.E = ^a;

  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

fn TestConvertibleToEnumVsEnum(a: Cpp.ConvertibleToEnum, b: Cpp.E) {
  let _: Cpp.E = a & b;
  let _: Cpp.E = a | b;
  let _: Cpp.E = a ^ b;
  let _: Cpp.E = ^a;

  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

// TODO: Should we allow these? For consistency with the cases in the next
// split, we should not, but we currently do.
fn TestConversionOnBothOperands(a: Cpp.ConvertibleToEnum, b: Cpp.E) {
  let _: Cpp.E = a & b;
  let _: Cpp.E = a | b;
  let _: Cpp.E = a ^ b;
  let _: Cpp.E = ^a;

  let _: bool = a == b;
  let _: bool = a != b;
  let _: bool = a < b;
  let _: bool = a <= b;
  let _: bool = a > b;
  let _: bool = a >= b;
}

// --- fail_convert_all_operands.carbon

// We do not currently support calls to builtin candidates where all operands
// are converted from class type to a primitive type. This is done to match
// Carbon rules.

library "[[@TEST_NAME]]";

import Cpp inline '''c++
struct ConvertibleToInt {
  operator int() const;
};
struct ConvertibleToFloat {
  operator float() const;
};
''';

fn TestInteger(a: Cpp.ConvertibleToInt, b: Cpp.ConvertibleToInt) {
  // Test arithmetic.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.AddWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `+` requires an impl of `Core.AddWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a + b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.SubWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `-` requires an impl of `Core.SubWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a - b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.MulWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `*` requires an impl of `Core.MulWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a * b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.DivWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `/` requires an impl of `Core.DivWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a / b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.ModWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `%` requires an impl of `Core.ModWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a % b;

  // Test bitwise.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.BitAndWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `&` requires an impl of `Core.BitAndWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a & b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.BitOrWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `|` requires an impl of `Core.BitOrWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a | b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.BitXorWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `^` requires an impl of `Core.BitXorWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a ^ b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18-19: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.LeftShiftWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18-19: note: `<<` requires an impl of `Core.LeftShiftWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:21: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a << b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18-19: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.RightShiftWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18-19: note: `>>` requires an impl of `Core.RightShiftWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:21: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: i32 = a >> b;

  // Test comparison.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20-21: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.EqWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20-21: note: `==` requires an impl of `Core.EqWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a == b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20-21: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.EqWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20-21: note: `!=` requires an impl of `Core.EqWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a != b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.OrderedWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20: note: `<` requires an impl of `Core.OrderedWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:22: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a < b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20-21: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.OrderedWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20-21: note: `<=` requires an impl of `Core.OrderedWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a <= b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.OrderedWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20: note: `>` requires an impl of `Core.OrderedWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:22: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a > b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20-21: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.OrderedWith(Cpp.ConvertibleToInt)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20-21: note: `>=` requires an impl of `Core.OrderedWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToInt` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.ConvertibleToInt` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a >= b);

  // Test unary.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.Negate` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: `-` requires an impl of `Core.Negate` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:17: note: operand has type `Cpp.ConvertibleToInt` [UnaryOperatorOperandType]
  // CHECK:STDERR:
  let _: i32 = -a;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: type `Cpp.ConvertibleToInt` does not implement interface `Core.BitComplement` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: `^` requires an impl of `Core.BitComplement` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:17: note: operand has type `Cpp.ConvertibleToInt` [UnaryOperatorOperandType]
  // CHECK:STDERR:
  let _: i32 = ^a;
}

fn TestFloat(a: Cpp.ConvertibleToFloat, b: Cpp.ConvertibleToFloat) {
  // Test arithmetic.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.AddWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `+` requires an impl of `Core.AddWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: f32 = a + b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.SubWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `-` requires an impl of `Core.SubWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: f32 = a - b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.MulWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `*` requires an impl of `Core.MulWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: f32 = a * b;
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:18: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.DivWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:18: note: `/` requires an impl of `Core.DivWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:20: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: f32 = a / b;

  // Test comparison.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20-21: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.EqWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20-21: note: `==` requires an impl of `Core.EqWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a == b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20-21: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.EqWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20-21: note: `!=` requires an impl of `Core.EqWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a != b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.OrderedWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20: note: `<` requires an impl of `Core.OrderedWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:22: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a < b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20-21: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.OrderedWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20-21: note: `<=` requires an impl of `Core.OrderedWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a <= b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.OrderedWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20: note: `>` requires an impl of `Core.OrderedWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:22: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a > b);
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+5]]:20-21: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.OrderedWith(Cpp.ConvertibleToFloat)` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:20-21: note: `>=` requires an impl of `Core.OrderedWith` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:18: note: left operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorLhsType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:23: note: right operand has type `Cpp.ConvertibleToFloat` [BinaryOperatorRhsType]
  // CHECK:STDERR:
  let _: bool = (a >= b);

  // Test unary.
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+4]]:16: error: type `Cpp.ConvertibleToFloat` does not implement interface `Core.Negate` [MissingImplForType]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+3]]:16: note: `-` requires an impl of `Core.Negate` [OperatorInterface]
  // CHECK:STDERR: fail_convert_all_operands.carbon:[[@LINE+2]]:17: note: operand has type `Cpp.ConvertibleToFloat` [UnaryOperatorOperandType]
  // CHECK:STDERR:
  let _: f32 = -a;
}
