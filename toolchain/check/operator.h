// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
#define CARBON_TOOLCHAIN_CHECK_OPERATOR_H_

#include <optional>

#include "toolchain/check/context.h"
#include "toolchain/check/core_identifier.h"
#include "toolchain/parse/node_ids.h"
#include "toolchain/sem_ir/ids.h"

namespace Carbon::Check {

struct Operator {
  CoreIdentifier interface_name;
  llvm::ArrayRef<SemIR::InstId> interface_args_ref = {};
  CoreIdentifier op_name = CoreIdentifier::Op;
};

// Marks `inst_id` with `label`, which says what type that operand contributed
// to an operation that failed.
//
// An operand whose type is already an error has nothing to say about it, and
// saying `<error>` against the code only adds noise to a diagnostic that is
// downstream of one already reported, so nothing is marked in that case.
inline auto AttachOperandType(Context& context, DiagnosticBuilder& builder,
                              SemIR::InstId inst_id,
                              const Diagnostics::LabelBase<TypeOfInstId>& label)
    -> void {
  if (inst_id.has_value() &&
      context.insts().Get(inst_id).type_id() != SemIR::ErrorInst::TypeId) {
    builder.Attach(inst_id, label, inst_id);
  }
}

// Marks the syntax at `loc_id` as what required an `impl`, naming the interface
// it looks for.
//
// Which interface a piece of syntax needs is a language rule the reader may not
// know, and the message reports only that an interface is missing, so this is
// where the two are tied together.
inline auto AttachOperatorSyntax(DiagnosticBuilder& builder,
                                 LocIdForDiagnostics loc_id,
                                 llvm::StringRef syntax,
                                 CoreIdentifier interface_name) -> void {
  CARBON_DIAGNOSTIC_LABEL(OperatorInterface, Primary,
                          "`{0}` requires an impl of `Core.{1}`", std::string,
                          std::string);
  builder.Attach(loc_id, OperatorInterface, syntax.str(),
                 interface_name.name().str());
}

// How a failed `impl` lookup for an operation is reported, which the caller
// supplies because it follows the syntax the operation was written in while the
// diagnostic is emitted several layers down.
struct MissingImplDiagnostic {
  // Where the message points, when the syntax has somewhere better than the
  // whole expression: the operator an infix expression is written with, the
  // index an indexing expression failed on. Unset points at the expression.
  std::optional<LocIdForDiagnostics> loc_id = std::nullopt;
  // Attaches the labels marking the operands and saying what each contributed.
  // The words follow the syntax too -- an infix operator has a left and a right
  // operand, while `a[i]` has an object and an index.
  DiagnosticAnnotateFn annotate = nullptr;
};

// Checks and builds SemIR for a unary operator expression. For example,
// `*operand` or `operand*`.
//
// On failure, an ErrorInst is returned and a diagnostic is produced unless
// `diagnose` is false. It is incorrect to specify `diagnose` as false if the
// resulting ErrorInst may appear in the produced SemIR.
//
// If specified, `missing_impl_diagnostic_context` is used to provide context
// for the diagnostic if the impl lookup for the operator fails, and
// `missing_impl` shapes that diagnostic.
auto BuildUnaryOperator(
    Context& context, SemIR::LocId loc_id, Operator op,
    SemIR::InstId operand_id, bool diagnose = true,
    DiagnosticContextFn missing_impl_diagnostic_context = nullptr,
    MissingImplDiagnostic missing_impl = {}) -> SemIR::InstId;

// Checks and builds SemIR for a binary operator expression. For example,
// `lhs_id * rhs_id`.
//
// On failure, an ErrorInst is returned and a diagnostic is produced unless
// `diagnose` is false. It is incorrect to specify `diagnose` as false if the
// resulting ErrorInst may appear in the produced SemIR.
//
// If specified, `missing_impl_diagnostic_context` is used to provide context
// for the diagnostic if the impl lookup for the operator fails, and
// `missing_impl` shapes that diagnostic.
auto BuildBinaryOperator(
    Context& context, SemIR::LocId loc_id, Operator op, SemIR::InstId lhs_id,
    SemIR::InstId rhs_id, bool diagnose = true,
    DiagnosticContextFn missing_impl_diagnostic_context = nullptr,
    MissingImplDiagnostic missing_impl = {}) -> SemIR::InstId;

}  // namespace Carbon::Check

#endif  // CARBON_TOOLCHAIN_CHECK_OPERATOR_H_
