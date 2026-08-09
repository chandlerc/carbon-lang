// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/lex/token_kind.h"
#include "toolchain/parse/context.h"
#include "toolchain/parse/handle.h"

namespace Carbon::Parse {

auto HandleIfExprFinishCondition(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExprIf, state.token, state.has_error);

  if (context.PositionIs(Lex::TokenKind::Then)) {
    context.PushState(StateKind::IfExprFinishThen);
    context.ConsumeChecked(Lex::TokenKind::Then);
    context.PushStateForExpr(*PrecedenceGroup::ForLeading(Lex::TokenKind::If));
  } else {
    CARBON_DIAGNOSTIC(ExpectedThenAfterIf, Error,
                      "expected `then` after `if` condition");
    CARBON_DIAGNOSTIC_LABEL(ThenGoesAfter, Primary,
                            "expected `then` after this token");
    CARBON_DIAGNOSTIC_LABEL(InIfExpr, Info, "in this `if` expression");
    if (!state.has_error) {
      context.emitter()
          .Build(*(context.position() - 1), ExpectedThenAfterIf)
          .Attach(*(context.position() - 1), ThenGoesAfter)
          .Attach(state.token, InIfExpr)
          .Emit();
    }
    // Add invalid nodes to substitute for `IfExprThen` and the final `Expr`.
    context.AddInvalidParse(*context.position());
    context.AddInvalidParse(*context.position());
    context.ReturnErrorOnState();
  }
}

auto HandleIfExprFinishThen(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExprThen, state.token, state.has_error);

  if (context.PositionIs(Lex::TokenKind::Else)) {
    context.PushState(StateKind::IfExprFinishElse);
    context.ConsumeChecked(Lex::TokenKind::Else);
    context.PushStateForExpr(*PrecedenceGroup::ForLeading(Lex::TokenKind::If));
  } else {
    CARBON_DIAGNOSTIC(ExpectedElseAfterIf, Error,
                      "expected `else` after `if ... then ...`");
    CARBON_DIAGNOSTIC_LABEL(ElseGoesAfter, Primary,
                            "expected `else` after this token");
    CARBON_DIAGNOSTIC_LABEL(ThenNeedsElse, Info, "this `then` needs an `else`");
    if (!state.has_error) {
      context.emitter()
          .Build(*(context.position() - 1), ExpectedElseAfterIf)
          .Attach(*(context.position() - 1), ElseGoesAfter)
          .Attach(state.token, ThenNeedsElse)
          .Emit();
    }
    // Add an invalid node to substitute for the final `Expr`.
    context.AddInvalidParse(*context.position());
    context.ReturnErrorOnState();
  }
}

auto HandleIfExprFinishElse(Context& context) -> void {
  auto else_state = context.PopState();

  // Propagate the location of `else`.
  auto if_state = context.PopState();
  if_state.token = else_state.token;
  if_state.has_error |= else_state.has_error;
  context.PushState(if_state);
}

auto HandleIfExprFinish(Context& context) -> void {
  auto state = context.PopState();

  context.AddNode(NodeKind::IfExprElse, state.token, state.has_error);
}

}  // namespace Carbon::Parse
