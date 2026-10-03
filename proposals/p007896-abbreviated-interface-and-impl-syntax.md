# Abbreviated `interface` and `impl` syntax

<!--
Part of the Carbon Language project, under the Apache License v2.0 with LLVM
Exceptions. See /LICENSE for license information.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

[Pull request](https://github.com/carbon-language/carbon-lang/pull/7896)

<!-- toc -->

## Table of contents

-   [Abstract](#abstract)
-   [Problem](#problem)
-   [Background](#background)
-   [Proposal](#proposal)
    -   [Anonymous primary functions](#anonymous-primary-functions)
    -   [Abbreviated `interface` declarations](#abbreviated-interface-declarations)
    -   [Abbreviated `impl` declarations and associated constant deduction](#abbreviated-impl-declarations-and-associated-constant-deduction)
    -   [Calling and accessing primary functions](#calling-and-accessing-primary-functions)
    -   [Updating `Op` interfaces](#updating-op-interfaces)
-   [Details](#details)
    -   [Interface extension and named constraints](#interface-extension-and-named-constraints)
    -   [Out-of-line definitions](#out-of-line-definitions)
-   [Future work](#future-work)
    -   [Defaults for return-type associated constants in abbreviated interfaces](#defaults-for-return-type-associated-constants-in-abbreviated-interfaces)
    -   [Deducing interface parameters from the primary function signature](#deducing-interface-parameters-from-the-primary-function-signature)
-   [Rationale](#rationale)
-   [Alternatives considered](#alternatives-considered)
    -   [Keep placeholder names like `Op` or repeat the interface name](#keep-placeholder-names-like-op-or-repeat-the-interface-name)
    -   [Only abbreviate `impl` declarations](#only-abbreviate-impl-declarations)
    -   [Make the implicit function name visible to unqualified lookup](#make-the-implicit-function-name-visible-to-unqualified-lookup)
    -   [Call primary functions without naming the member](#call-primary-functions-without-naming-the-member)
    -   [Deduce associated constants in braced `impl` definitions](#deduce-associated-constants-in-braced-impl-definitions)
    -   [Dedicated `op` keyword](#dedicated-op-keyword)

<!-- tocstop -->

## Abstract

This proposal introduces **anonymous primary functions** in `interface` and
`impl` declarations, along with **abbreviated `interface` and `impl` syntax**
that omits the enclosing `{ ... }` braces when an interface or implementation
consists of a single primary function:

```carbon
interface Negate fn (self) -> (Result: type);

interface AddWith(U: type) {
  default let Result: type = Self;
  fn (self, other: U) -> Result;
}

class Point {
  var x: i32;
  var y: i32;

  impl as Negate fn (self) -> Point {
    return {.x = -self.x, .y = -self.y};
  }
  impl as AddWith(Point) fn (self, other: Point) -> Point {
    return {.x = self.x + other.x, .y = self.y + other.y};
  }
}
```

An anonymous `fn` declaration in an `interface` implicitly takes the name of the
enclosing `interface` for qualified member lookup (such as `p.Negate()`,
`p.AddWith(q)`, or `Point.(AddWith(Point).AddWith)`), and can also be referenced
directly through the interface or constraint in compound member access
(`p.(AddWith(Point))(q)` and `Point.impl(AddWith(Point))`). In an abbreviated
`interface`, the return type may declare an associated constant in-line using
`-> (Result: type)`. In an abbreviated `impl`, required associated constants of
the interface are automatically deduced from the primary function's signature.

This replaces placeholder method names like `Op` across operator, callable, and
other single-operation interfaces whose names work naturally as method names on
implementing types.

## Problem

Many interfaces in Carbon represent a single operation: arithmetic and bitwise
operators (`AddWith`, `Negate`), assignment and compound assignment
(`AssignWith`, `AddAssignWith`, `Inc`, `Dec`), callables (`Call`), pointer
dereferencing (`Deref`), and pattern matching (`Match`). Previously, declaring
and implementing these interfaces had several sources of friction and
redundancy:

1.  **Placeholder or redundant function names**: Because every associated
    function in an interface required an explicit identifier, single-function
    interfaces either used an uninformative placeholder name like `Op` or
    repeated the interface's name (`interface EqualWith(U: type) { fn
    EqualWith... }`). When an interface was implemented with `extend impl as` in
    a class, a placeholder name like `Op` could not be usefully extended into
    the class's namespace without colliding with other interfaces using `Op` or
    exposing a meaningless `x.Op()` method. Conversely, repeating the interface
    name in both the `interface` and `impl` declarations was verbose and could
    trigger name lookup hazards inside the interface body.
2.  **Ceremony for single-function `interface` and `impl` declarations**:
    Implementing a single-operation interface with an associated `Result` type
    required both a `where .Result = ...` clause on the facet type and braces
    around a single `fn` definition:

    ```carbon
    // Previous syntax:
    impl Point as AddWith(Point) where .Result = Point {
      fn Op[self: Self](other: Point) -> Point {
        return {.x = self.x + other.x, .y = self.y + other.y};
      }
    }
    ```

    Here, the return type `Point` had to be written twice—once in
    `where .Result = Point` and again in `-> Point`—and programmers writing a
    simple operator implementation had to learn associated type rewrite
    constraints before they could return a type other than `Self`.

## Background

-   [Issue #4711: Abbreviated `interface` and `impl` syntax](https://github.com/carbon-language/carbon-lang/issues/4711)
    tracked the design of a streamlined syntax for single-function interfaces
    and their implementations.
-   Earlier operator proposals (including
    [#1083 (arithmetic)](/proposals/p001083-arithmetic-expressions.md),
    [#1178 (operator interfaces)](/proposals/p001178-rework-operator-interfaces.md),
    [#1191 (bitwise)](/proposals/p001191-bitwise-and-shift-operators.md),
    [#2511 (assignment)](/proposals/p002511-assignment-statements.md),
    [#2875 (functions and `Call`)](/proposals/p002875-functions-function-types-and-function-calls.md),
    and
    [#3720 (member binding)](/proposals/p003720-member-binding-operators.md))
    adopted `fn Op` as a temporary convention for operator interfaces while
    explicitly noting that `Op` was a placeholder pending a cleaner mechanism.
-   [Proposal #3848: Lambdas](/proposals/p003848-lambdas.md) established
    continuous `fn` syntax across named functions and anonymous functions
    (lambdas), where omitting the function name after `fn` introduces an
    anonymous function.
-   [Proposal #5253: Redefining `alias` syntax](https://github.com/carbon-language/carbon-lang/pull/5253)
    and
    [Proposal #5337: Interface extension and `final impl` update](/proposals/p005337-interface-extension-and-final-impl-update.md)
    refined how interfaces and named constraints extend and alias members of
    other interfaces.
-   [Proposal #5366: The name of an `impl` in `class` scope](/proposals/p005366-the-name-of-an-impl-in-class-scope.md)
    defined how `impl` declarations inside a class are named and redeclared
    out-of-line as `impl Class.(as Interface)`.
-   [Proposal #7016: Parameter binding simplifications](https://github.com/carbon-language/carbon-lang/pull/7016)
    and
    [Proposal #7697: Implicit compile-time binding in `interface`, `constraint`, and `impl`](https://github.com/carbon-language/carbon-lang/pull/7697)
    simplified method receivers (`fn (self, ...)` and `fn (ref self, ...)`) and
    compile-time bindings in generic declarations.
-   The design in this proposal was developed in
    [Dedicated syntax for interface functions](https://docs.google.com/document/d/1l3AdRSEbC7uWU6eRzRLu47IXF83-PR6f7tBz_Rqb2EY/edit)
    and discussed in open discussions on
    [2025-03-28](https://docs.google.com/document/d/1z6A301lu272676NA7AA03C1sKF3129c1YG4e1s32i7g/edit#heading=h.x8bbebuxbdpt)
    and
    [2025-04-30](https://docs.google.com/document/d/1z6A301lu272676NA7AA03C1sKF3129c1YG4e1s32i7g/edit#heading=h.zbe2kfk3yplq).

## Proposal

### Anonymous primary functions

An `interface` may contain at most one anonymous `fn` declaration—a function
declaration that omits the function name between `fn` and its parameter list:

```carbon
interface AddWith(U: type) {
  default let Result: type = Self;
  fn (self, other: U) -> Result;
}
```

This function is the **primary function** of the interface:

-   It implicitly has the same name as the enclosing `interface` for **qualified
    member lookup** (such as `x.AddWith(y)`, `Self.AddWith(y)`, or
    `T.(AddWith(U).AddWith)`).
-   It is **not** visible to **unqualified name lookup** inside the `interface`,
    an extending `interface` or `constraint`, or an `impl`. Within those scopes,
    the unqualified identifier `AddWith` continues to refer to the interface
    itself, avoiding self-shadowing ("name poisoning"). To call the primary
    function from within the interface or an `impl`, use qualified lookup such
    as `self.AddWith(other)` or `Self.AddWith(self, other)`.

Correspondingly, an `impl` of an interface with a primary function may omit the
function name on at most one `fn` declaration inside the `impl` body, which
implements the interface's primary function. It may also write the interface's
name explicitly (`fn AddWith(self, other: U) -> Self`).

### Abbreviated `interface` declarations

When an `interface` only declares a single primary function (and optionally an
associated type or constant for its return type), the `{ ... }` braces around
the interface body may be omitted:

```carbon
interface Inc fn (ref self);
```

If the primary function's return type in an abbreviated `interface` is written
as a parenthesized binding pattern `-> (Name: Type)`, it declares an associated
constant on the interface and uses that constant as the function's return type.
For example:

```carbon
interface Deref fn (self) -> (Result: type);
interface Call(... each Arg: type)
    fn (self, ... each arg: each Arg) -> (Result: type);
```

is completely equivalent to the braced form:

```carbon
interface Deref {
  let Result: type;
  fn (self) -> Result;
}
```

To keep the abbreviated syntax simple, specifying a `default` value for the
return-type associated constant (such as `default let Result: type = Self;` in
`AddWith`) requires the braced `interface { ... }` form.

### Abbreviated `impl` declarations and associated constant deduction

An `impl` declaration (including `extend impl` and `final impl`) may omit the
`{ ... }` braces and follow the facet type directly with an anonymous `fn`
declaration or definition:

```carbon
class CustomInt {
  var value: i32;

  extend impl as AddWith(CustomInt)
      fn (self, other: CustomInt) -> CustomInt =>
          {.value = self.value + other.value};

  extend impl as Inc fn (ref self) {
    ++self.value;
  }
}
```

Abbreviated `impl` syntax is valid for any interface that has a primary function
when all other requirements of the interface either have defaults or can be
deduced from the primary function's signature, whether the `interface` itself
was declared using abbreviated or braced syntax.

In an abbreviated `impl` declaration, any required (non-`default`, non-`final`)
associated constants of the interface—and any associated constants without
defaults that appear in the interface's primary function signature, such as
`Result` in `AddWith`—are **automatically deduced** by matching the `impl`'s
`fn` signature against the interface's primary `fn` signature:

-   In `impl as AddWith(CustomInt) fn (self, other: CustomInt) -> CustomInt`,
    matching `-> CustomInt` against `-> Result` deduces `.Result = CustomInt`
    without requiring an explicit `where .Result = CustomInt` clause.
-   Deduced associated constants belong to the `impl`, so they cannot depend on
    generic parameters of the `fn` itself (though they may depend on `forall`
    parameters of the `impl` or enclosing generic scopes).
-   An abbreviated `impl` forward declaration (ending in `;` instead of a body)
    performs the same deduction, making the deduced associated constants
    available immediately at the forward declaration site.
-   Associated constant deduction **only** occurs in the abbreviated `impl ...
    fn ...` form. Once `{ ... }` braces are used for an `impl`, associated
    constants must be specified by way of `where` constraints (or use their
    defaults).

### Calling and accessing primary functions

Because a primary function has the name of its enclosing `interface` for
qualified lookup, it can be called or accessed like any named interface method:

-   As a method on a value when extended into a class (`extend impl as`) or on a
    generic type constrained by the interface: `x.AddWith(y)`, `x.Negate()`,
    `f.Call(a, b)`.
-   By way of qualified compound member access: `x.(AddWith(U).AddWith)(y)` or
    `T.(AddWith(U).AddWith)(x, y)`.
-   By way of `impl` member access: `MyType.impl(AddWith(U)).AddWith`.

In addition, when a facet type `I` in `x.(I)` or `T.impl(I)` has a primary
function (or an anonymous alias), omitting the redundant `.I` member name is
permitted and designates the primary function:

-   `x.(AddWith(U))(y)` is equivalent to `x.(AddWith(U).AddWith)(y)`.
-   `T.(AddWith(U))(x, y)` is equivalent to `T.(AddWith(U).AddWith)(x, y)`.
-   `MyType.impl(AddWith(U))` in a value/function context designates
    `MyType.impl(AddWith(U)).AddWith`.

This shorthand is unambiguous because a facet type itself is not an associated
entity or instance member.

### Updating `Op` interfaces

Whether an existing single-function interface switches to an anonymous primary
function is governed by a simple rule: **use a primary function when the
interface name itself works well as a method name on a type implementing the
interface**.

-   **Switch from `Op` to an anonymous primary function**:
    -   Arithmetic interfaces: `Negate`, `AddWith`, `SubWith`, `MulWith`,
        `DivWith`, `ModWith`.
    -   Bitwise interfaces: `BitComplement`, `BitAndWith`, `BitOrWith`,
        `BitXorWith`, `LeftShiftWith`, `RightShiftWith`.
    -   Assignment and increment/decrement interfaces: `AssignWith`,
        `AddAssignWith`, `SubAssignWith`, `MulAssignWith`, `DivAssignWith`,
        `ModAssignWith`, `BitAndAssignWith`, `BitOrAssignWith`,
        `BitXorAssignWith`, `LeftShiftAssignWith`, `RightShiftAssignWith`,
        `Inc`, `Dec`.
    -   Callable interface: `Call`.
    -   Pointer dereference interface: `Deref`.
    -   Member binding interfaces: `BindToValue`, `BindToRef`.
    -   Sum type matching and copying interfaces in design docs: `Match`,
        `Copy`.
-   **Keep an explicit method name**:
    -   Conversion interfaces (`As`, `ImplicitAs`, `ReferenceImplicitAs`) keep
        `fn Convert`, because `x.As()` is not a clear method name on an
        implementing type.
    -   Comparison interfaces (`EqWith`, `OrderedWith`) already have multiple
        methods (`Equal`, `NotEqual`, `Compare`) with distinct names and are
        unchanged.

## Details

Full specification details are integrated directly into the design documentation
in this pull request:

-   [`/docs/design/generics/overview.md`](/docs/design/generics/overview.md) and
    [`/docs/design/generics/details.md`](/docs/design/generics/details.md)
-   [`/docs/design/expressions/member_access.md`](/docs/design/expressions/member_access.md)
-   [`/docs/design/expressions/arithmetic.md`](/docs/design/expressions/arithmetic.md)
    and
    [`/docs/design/expressions/bitwise.md`](/docs/design/expressions/bitwise.md)
-   [`/docs/design/assignment.md`](/docs/design/assignment.md) and
    [`/docs/design/functions.md`](/docs/design/functions.md)

A few key interactions are summarized below.

### Interface extension and named constraints

When an `interface` or `constraint` extends another interface with a primary
function, the two extension mechanisms behave according to their existing models
from [#5337](/proposals/p005337-interface-extension-and-final-impl-update.md):

-   **`extend require impls I`**: Aliases `I`'s members into the extending scope
    under their qualified names. The primary function of `I` is brought in with
    the qualified name `I`, **not** renamed to the extending interface or
    constraint's name. This allows a constraint or interface to extend multiple
    single-function interfaces (such as both `AddWith(Self)` and
    `SubWith(Self)`) without their primary functions colliding.
-   **`extend [final] impl as I` in `interface J`**: Copies `I`'s members to
    form new members of `J` and generates a blanket `impl` of `I` forwarding to
    `J`. An anonymous primary function in `I` is copied as `J`'s anonymous
    primary function (taking the qualified name `J`), and the generated blanket
    `impl` forwards `I`'s primary function to `J`'s primary function. (If `J`
    uses `extend impl as` with multiple interfaces that have primary functions,
    `J` does not automatically acquire a primary function unless disambiguated.)
-   **Anonymous `alias` declarations (`alias = Target;`)**: An `interface` or
    `constraint` may declare at most one anonymous alias by omitting the alias
    name before `=`. Like an anonymous primary function, an anonymous alias
    implicitly takes the enclosing `interface` or `constraint`'s name for
    qualified lookup only, and can be accessed by way of `x.(Constraint)` when
    it aliases a function. This allows named constraints for operators to expose
    a convenient primary function name matching the constraint:

    ```carbon
    constraint Add {
      extend require impls AddWith(Self) where .Result = Self;
      alias = AddWith(Self).AddWith;
    }
    ```

    For a type parameter `T: Add` and values `x: T, y: T`, callers can write
    `x.Add(y)`, `x.AddWith(y)`, or `x.(Add)(y)`.

### Out-of-line definitions

Out-of-line definitions of primary functions are supported in three forms:

1.  **Out-of-line abbreviated `impl` definition** (following
    [#5366](/proposals/p005366-the-name-of-an-impl-in-class-scope.md)):

    ```carbon
    class MyType {
      impl as AddWith(OtherType);
    }

    impl MyType.(as AddWith(OtherType))
        fn (self, other: OtherType) -> MyType { ... }
    ```

2.  **Standalone `fn` definition naming `.InterfaceName`** (works for both
    `impl` functions and `interface` default functions):

    ```carbon
    fn (MyType as AddWith(OtherType)).AddWith(
        self, other: OtherType) -> MyType { ... }

    fn MyType.(as AddWith(OtherType)).AddWith(
        self, other: OtherType) -> MyType { ... }

    fn AddWith(U: type).AddWith(self, other: U) -> Result { ... }
    ```

3.  **Standalone `fn` definition on an `impl` omitting `.InterfaceName`**:
    Because the `impl` scope in `(MyType as Interface)` or
    `MyType.(as Interface)` is enclosed in parentheses, the `.InterfaceName`
    suffix can be omitted unambiguously before the function parameter list:

    ```carbon
    fn (MyType as AddWith(OtherType))(
        self, other: OtherType) -> MyType { ... }

    fn MyType.(as AddWith(OtherType))(
        self, other: OtherType) -> MyType { ... }
    ```

    (By contrast, an out-of-line `interface` default function definition still
    requires `.InterfaceName`, as in `fn AddWith(U: type).AddWith(...)`, to
    separate the interface's parameter list from the function's parameter list.)

## Future work

### Defaults for return-type associated constants in abbreviated interfaces

Under this proposal, declaring a default value for a return-type associated
constant (such as `default let Result: type = Self;`) requires writing a braced
`interface { ... }` body rather than using the abbreviated `interface` syntax.
Future work on generalized default value syntax in patterns may provide a
natural spelling—such as `-> (Result: type = Self)`—that would allow interfaces
like `AddWith` to also use the abbreviated `interface` form.

### Deducing interface parameters from the primary function signature

This proposal still requires interface parameters to be listed explicitly on the
interface name in abbreviated declarations, such as
`interface AssignWith(U: type) fn (ref self, other: U);`. Future work may
explore whether interface parameters could be deduced or declared directly in
the primary function's parameter list in some cases.

## Rationale

This proposal advances several of Carbon's goals and design principles:

-   **[Progressive disclosure](/docs/project/principles/progressive_disclosure.md)**:
    Programmers implementing a single-function interface (such as an overloaded
    operator or `Call`) can write a direct `impl ... fn (...) -> ReturnType`
    without first needing to learn associated type rewrite syntax
    (`where .Result = ReturnType`). At the same time, the abbreviated form is
    not a "lie-to-children": it desugars directly to the general `interface` and
    `impl` model when more advanced features (such as multiple members or
    explicit `where` clauses) are needed.
-   **[Code that is easy to read, understand, and write](/docs/project/goals.md#code-that-is-easy-to-read-understand-and-write)**:
    Eliminating placeholder `Op` names, redundant braces, and duplicated return
    types makes both interface definitions and implementations significantly
    more concise while giving methods meaningful names (`x.AddWith(y)`,
    `x.Negate()`, `f.Call(...)`) when extended into classes or used in generic
    code.
-   **[Language tools and ecosystem](/docs/project/goals.md#language-tools-and-ecosystem)**:
    Restricting the implicit function name to qualified lookup avoids
    context-sensitive name collisions inside interface bodies, keeping name
    resolution predictable for both humans and tools.

## Alternatives considered

### Keep placeholder names like `Op` or repeat the interface name

We could keep the status quo where every interface function has an explicit
identifier—either a generic placeholder like `Op` or a repetition of the
interface name.

However, `Op` works poorly with `extend impl as` in a class: extending
`AddWith(Point)` into `Point` would inject a method named `Op` into `Point`,
colliding with any other extended operator interface and producing an unhelpful
`p.Op(q)` method name. Requiring every single-function interface and `impl` to
explicitly write out the interface name twice (`interface Negate { fn Negate...
}`, `impl as Negate { fn Negate... }`) is repetitive and introduces unqualified
lookup shadowing inside the interface scope unless special lookup rules are
added anyway.

### Only abbreviate `impl` declarations

We considered keeping all `interface` declarations braced (`interface Inc { fn
(ref self); }`) and only introducing the brace-omitting abbreviation on `impl`
declarations, on the grounds that `impl`s are written much more frequently than
`interface`s.

However, single-function interfaces are common enough in both standard and
user-defined libraries that the abbreviated `interface` form provides valuable
symmetry with `impl`—particularly for interfaces without associated types (like
`Inc` and `AssignWith`) and interfaces with an un-defaulted return type (like
`Deref` and `Call`).

### Make the implicit function name visible to unqualified lookup

If an anonymous primary function inside `interface I` also introduced `I` into
unqualified lookup within the body of `I` (or within an extending `interface` or
`constraint`), any unqualified mention of `I`—for example, referring to `I` with
different arguments or in a constraint—would find the function instead of the
interface. By making the implicit name visible only to **qualified** member
lookup (`Self.I`, `x.I`, `T.(I.I)`), unqualified `I` unambiguously refers to the
interface itself.

### Call primary functions without naming the member

We considered allowing a value of a type constrained by a single-function
interface `I` to be called directly as `x(y)` or with a placeholder member
syntax such as `x._(y)`.

Calling `x(y)` directly would conflict with the `Call` interface when a type
implements both `Call` and another single-function interface (or when a generic
parameter is constrained by multiple interfaces). Using `x._(y)` would still
fail to disambiguate when two extended interfaces in a constraint both have
primary functions. Giving the primary function the qualified name of its
interface (`x.I(y)`) and supporting `x.(I)(y)` provides a meaningful name at the
call site and naturally disambiguates multiple interfaces.

### Deduce associated constants in braced `impl` definitions

We considered also deducing associated constants like `Result` from the primary
function signature inside a braced `impl` body:

```carbon
impl Point as AddWith(Point) {
  fn (self, other: Point) -> Point { ... }
}
```

We decided against this because once an `impl` uses `{ ... }` braces, it may
contain multiple declarations, aliases, or out-of-line forward declarations, and
making the facet type of the `impl` implicitly depend on declarations inside the
braces is less clear than either using the single-declaration abbreviated form
(`impl Point as AddWith(Point) fn ...`) or writing `where .Result = Point` on
the `impl` header.

### Dedicated `op` keyword

Early discussions in
[#4711](https://github.com/carbon-language/carbon-lang/issues/4711) explored
using an `op` keyword in place of `fn` for operator interfaces and
implementations. However, single-function interfaces are not limited to built-in
operators (for example, `Call`, `Match`, `Copy`, or user-defined single-action
interfaces), and reusing `fn` with an omitted name aligns directly with Carbon's
existing syntax for anonymous functions (lambdas) from
[#3848](/proposals/p003848-lambdas.md) without introducing a new declaration
keyword.
