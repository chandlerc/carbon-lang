// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// INCLUDE-FILE: toolchain/testing/testdata/min_prelude/none.carbon
// ARGS: --include-diagnostic-kind --diagnostic-snippets --terminal-unicode=always compile --phase=parse %s
//
// AUTOUPDATE
// TIP: To test this file alone, run:
// TIP:   bazel test //toolchain/testing:file_test --test_arg=--file_tests=toolchain/diagnostics/testdata/fail_recovery_demo.carbon
// TIP: To dump output, run:
// TIP:   bazel run //toolchain/testing:file_test -- --dump_output --file_tests=toolchain/diagnostics/testdata/fail_recovery_demo.carbon

// A tour of how a missing token is drawn, which is the case where nothing in the
// source is wrong: what the reader needs is not a range to look at but a place
// to put something. See /toolchain/docs/diagnostics_rendering.md for the
// rendering and /toolchain/docs/diagnostics.md for the convention.
//
// The rule is that the range marks the token the reader wrote that the
// missing one goes after, and the label says so. The mark lands on the last
// thing the developer typed rather than on whatever turned up instead -- at
// the end of a file, that difference is between marking code and marking
// nothing at all. For an opening delimiter that token is the introducer that
// called for it, which the message already names.
//
// This is a parse-phase tour, so nothing here reaches checking. What each line
// is for:
//
// - `fn Paint`: a separator missing between two elements of a list, marked on
//   the element it goes after.
// - `if true`: an opening delimiter missing after its introducer, marked on
//   the introducer the message names.
// - `var n`: the lexer inserting a `)` to recover, which the label reports,
//   because everything reported after that point is about that reading of the
//   file rather than about what was written.
// - `class Widget`: a `;` missing at the end of a declaration, marked on the
//   last token of the declaration. This is last in the file because recovery
//   skips to what looks like the end of the declaration and so eats whatever
//   follows -- and it is the case where marking the token that turned up
//   instead would have marked the end of the file.

library "recovery_demo";

// CHECK:STDERR: error: expected `,` or `)` [UnexpectedTokenAfterListElement]
// CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+7]]:13
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->  52 │ fn Paint(a: i32 b: i32);
// CHECK:STDERR:        ·             ─┬─
// CHECK:STDERR:        ·              ╰──┤ expected `,` or `)` after this token [ListSeparatorGoesAfter]
// CHECK:STDERR:        ╰────────────
// CHECK:STDERR:
fn Paint(a: i32 b: i32);

fn Blend() {
  // CHECK:STDERR: error: expected `(` after `if` [ExpectedParenAfter]
  // CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+7]]:3
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->  63 │   if true { }
  // CHECK:STDERR:        ·   ─┬
  // CHECK:STDERR:        ·    ╰──┤ expected `(` after this token [ParenGoesAfter]
  // CHECK:STDERR:        ╰────────────
  // CHECK:STDERR:
  if true { }
}

fn Tint() {
  // CHECK:STDERR: error: opening symbol without a corresponding closing symbol [UnmatchedOpening]
  // CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+7]]:16
  // CHECK:STDERR:        ┆
  // CHECK:STDERR: ->  75 │   var n: i32 = (1;
  // CHECK:STDERR:        ·                ─ ┬
  // CHECK:STDERR:        ·                  ╰──┤ possibly missing `)` here [PossiblyMissingBracketHere]
  // CHECK:STDERR:        ╰────────────
  // CHECK:STDERR:
  var n: i32 = (1;
}

// CHECK:STDERR: error: `class` declarations must either end with a `;` or have a `{ ... }` block for a
// CHECK:STDERR:        definition
// CHECK:STDERR:        [ExpectedDeclSemiOrDefinition]
// CHECK:STDERR:        ╭─┤ fail_recovery_demo.carbon:[[@LINE+7]]:7
// CHECK:STDERR:        ┆
// CHECK:STDERR: ->  88 │ class Widget
// CHECK:STDERR:        ·       ───┬──
// CHECK:STDERR:        ·          ╰──┤ expected `;` or `{` after this token [DeclSemiOrBlockGoesAfter]
// CHECK:STDERR:        ╰────────────
// CHECK:STDERR:
class Widget
