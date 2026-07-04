// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/benchmarking/source_gen.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <string>
#include <utility>

#include "common/raw_string_ostream.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "toolchain/lex/token_kind.h"

namespace Carbon::Testing {

auto SourceGen::Global() -> SourceGen& {
  static SourceGen global_gen;
  return global_gen;
}

SourceGen::SourceGen(Language language) : language_(language) {}

// Heuristic numbers used in synthesizing various identifier sequences.
static constexpr int MinClassNameLength = 5;
static constexpr int MinMemberNameLength = 4;

// Fixed length for an inline function definition's local-variable names. A
// local is re-emitted a position-dependent number of times by the body's chain
// of initializers, so a single fixed length keeps the byte total contributed by
// locals dependent only on how many there are, never on which are sampled. The
// length is also kept below `MinMemberNameLength`, so a local can never shadow
// the class names the body produces via `Make`.
static constexpr int LocalNameLength = 3;

// The shuffled state used to generate some number of classes.
//
// This state encodes everything used to generate class definitions. The state
// will be consumed until empty.
//
// Detailed comments for out-of-line methods are on their definitions.
class SourceGen::ClassGenState {
 public:
  ClassGenState(SourceGen& gen, int num_classes,
                const ClassParams& class_params,
                const TypeUseParams& type_use_params,
                bool define_decls_out_of_line);

  auto public_function_param_counts() -> llvm::SmallVectorImpl<int>& {
    return public_function_param_counts_;
  }
  auto public_method_param_counts() -> llvm::SmallVectorImpl<int>& {
    return public_method_param_counts_;
  }
  auto private_function_param_counts() -> llvm::SmallVectorImpl<int>& {
    return private_function_param_counts_;
  }
  auto private_method_param_counts() -> llvm::SmallVectorImpl<int>& {
    return private_method_param_counts_;
  }

  auto inline_function_param_counts() -> llvm::SmallVectorImpl<int>& {
    return inline_function_param_counts_;
  }
  auto local_counts() -> llvm::SmallVectorImpl<int>& { return local_counts_; }

  auto class_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return class_names_;
  }
  auto method_function_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return method_function_names_;
  }
  auto field_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return field_names_;
  }
  auto param_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return param_names_;
  }

  auto inline_function_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return inline_function_names_;
  }
  auto inline_param_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return inline_param_names_;
  }
  auto local_names() -> llvm::SmallVectorImpl<llvm::StringRef>& {
    return local_names_;
  }

  auto AddValidTypeName(llvm::StringRef type_name) -> void {
    valid_type_names_.Insert(type_name);
  }

  // The set of all declared class names, for excluding from identifier pools
  // so that names introduced into a scope (parameters, members, fields) can't
  // shadow a class name that a body relies on (for example via a `Make` call).
  auto class_name_set() -> const Set<llvm::StringRef>& {
    return class_name_set_;
  }

  // Emits an expression producing a value of the given type. For a class type
  // this is a call to that class's nested `Make` factory; for a builtin it is
  // the configured value expression. Only types that can be produced (classes
  // and builtins with a non-empty value expression) are ever passed here, which
  // the produced-position type pools guarantee.
  auto ProduceValue(llvm::StringRef type, bool is_cpp, llvm::raw_ostream& os)
      -> void {
    if (class_name_set_.Contains(type)) {
      os << type << (is_cpp ? "::Make()" : ".Make()");
    } else {
      os << fixed_value_.Lookup(type).value();
    }
  }

  // A single reference to a type in a pool: the type's spelling, and -- for
  // pools whose positions consume their value -- the consumer template
  // assigned to this particular use when the pool was built.
  struct TypeUse {
    llvm::StringRef name;
    llvm::StringRef consumer;
  };

  // Type-use accessors. Type uses are partitioned into pools by how many times
  // their type is emitted in the output -- the "emission profile" -- so that
  // every entry in a pool is emitted the same number of times and the total is
  // therefore independent of the shuffle. The profile depends on how many
  // signatures the use appears in (one for an in-class declaration or inline
  // definition; two for a declaration that is also defined out-of-line), on
  // whether the use is "produced" (a function return or field, whose value is
  // constructed in a body, re-emitting class types via `Make`), and on whether
  // it is "consumed" (a parameter of a function with a body, whose value is
  // read at a cost fixed by the entry's assigned consumer template).

  // One signature, not produced, not consumed: the returns and parameters of
  // pure declarations (no out-of-line definitions), plus field types when
  // this file generates no bodies (see `GetFieldType`). Class-capable
  // (declaration returns are the self-absorbers).
  auto GetDeclType() -> TypeUse { return GetValidTypeUse(decl_type_pool_); }
  // One signature, consumed: inline-definition parameters. Each inline
  // definition has one guaranteed ("mirror") parameter, so every class has
  // `inline_function_defs` self-absorbing slots and the pool can admit class
  // types.
  auto GetInlineParamType() -> TypeUse {
    return GetValidTypeUse(inline_param_pool_);
  }
  // One signature, produced: inline-definition return types, plus field types
  // when this file generates bodies. Class-capable (inline returns are the
  // self-absorbers).
  auto GetProducedType() -> TypeUse {
    return GetValidTypeUse(produced_type_pool_);
  }
  // Two signatures, produced: out-of-line-definition return types.
  auto GetOutOfLineReturnType() -> TypeUse {
    return GetValidTypeUse(outofline_return_pool_);
  }
  // Two signatures (in-class declaration plus out-of-line definition),
  // consumed: out-of-line-definition parameters. Each out-of-line-defined
  // function is given one guaranteed extra ("mirror") parameter, so every
  // class has at least `decls_per_class` parameter slots that become valid
  // after the class itself does -- the self-absorbers that let this pool admit
  // class types with a nonzero cap.
  auto GetOutOfLineParamType() -> TypeUse {
    return GetValidTypeUse(outofline_param_pool_);
  }
  // Field types: `Make` constructs every field when this file generates
  // bodies, making the fields produced uses; in the pure-declaration pattern
  // no `Make` is emitted and a field type is spelled exactly once, the same
  // emission profile as the declaration returns and parameters. Sharing the
  // decl pool there also lets pure-declaration fields keep using class
  // references and non-producible types.
  auto GetFieldType() -> TypeUse {
    return GetValidTypeUse(generate_bodies_ ? produced_type_pool_
                                            : decl_type_pool_);
  }

  auto generates_bodies() -> bool { return generate_bodies_; }
  auto define_decls_out_of_line() -> bool { return split_type_pools_; }

  auto type_pools_empty() -> bool {
    return decl_type_pool_.uses.empty() && inline_param_pool_.uses.empty() &&
           produced_type_pool_.uses.empty() &&
           outofline_return_pool_.uses.empty() &&
           outofline_param_pool_.uses.empty();
  }

 private:
  // A pool of type references that is consumed as type uses are emitted.
  // `last_index` tracks the search position within `uses`.
  struct TypePool {
    llvm::SmallVector<TypeUse> uses;
    int last_index = 0;
  };

  auto GetValidTypeUse(TypePool& pool) -> TypeUse;

  auto BuildClassAndTypeNames(SourceGen& gen, int num_classes,
                              const ClassParams& class_params,
                              const TypeUseParams& type_use_params) -> void;
  // Builds one type pool of `num_types` references. `max_refs_per_class` caps
  // class references so the search always drains (see comment on the
  // definition). When `producible_only` is set, only builtins with a value
  // expression are used for the fixed-type portion, since these types appear in
  // produced positions (return and field types) that must construct a value.
  // When `consumed` is set, every entry is also assigned a consumer template,
  // round-robin from its type's configured list, before the pool is shuffled;
  // the template mix is therefore seed-independent even though which use gets
  // which template is not.
  auto BuildTypePool(SourceGen& gen, int num_types, int max_refs_per_class,
                     bool producible_only, bool consumed,
                     const TypeUseParams& type_use_params) -> TypePool;

  llvm::SmallVector<int> public_function_param_counts_;
  llvm::SmallVector<int> public_method_param_counts_;
  llvm::SmallVector<int> private_function_param_counts_;
  llvm::SmallVector<int> private_method_param_counts_;

  // Parameter and local-variable counts for inline-defined functions. These use
  // dedicated distributions so the counts -- and thus the totals re-emitted in
  // each body -- are independent of the random seed.
  llvm::SmallVector<int> inline_function_param_counts_;
  llvm::SmallVector<int> local_counts_;

  llvm::SmallVector<llvm::StringRef> class_names_;
  // Names for declared functions and methods are kept separate from field names
  // so that, when out-of-line definitions are generated, the function and
  // method names re-emitted there form a fully-consumed pool with a
  // seed-independent total size. Within a class, field names are kept distinct
  // from these via `UniqueIdentifierPopper::Reserve`.
  llvm::SmallVector<llvm::StringRef> method_function_names_;
  llvm::SmallVector<llvm::StringRef> field_names_;
  llvm::SmallVector<llvm::StringRef> param_names_;

  // Dedicated name pools for inline definitions' function names, parameters,
  // and body locals. Inline function names and locals use a length distribution
  // bounded below `MinMemberNameLength` so they can never collide with
  // declaration, field, or class names (which lets them stay deterministic
  // without cross-pool coordination): inline function names are emitted once,
  // so a varied length is fine, but they are not re-emitted out-of-line like
  // declared functions and so cannot share the declaration name pool. Parameter
  // names use the full length distribution. Local names keep a single fixed
  // length (see the construction of these pools for the details).
  llvm::SmallVector<llvm::StringRef> inline_function_names_;
  llvm::SmallVector<llvm::StringRef> inline_param_names_;
  llvm::SmallVector<llvm::StringRef> local_names_;

  // Type-reference pools, partitioned by emission profile (see the type-use
  // accessors). `generate_bodies_` records whether this file generates any
  // function bodies, which determines whether `Make` and `Checksum` are
  // emitted and which pool the field types use. `split_type_pools_` records
  // whether out-of-line definitions are generated, which determines whether
  // the declaration returns/params land in the decl pool (in-class
  // declarations only) or in the out-of-line pools (declared in-class and
  // defined out-of-line).
  bool generate_bodies_ = false;
  bool split_type_pools_ = false;
  TypePool decl_type_pool_;
  TypePool inline_param_pool_;
  TypePool produced_type_pool_;
  TypePool outofline_return_pool_;
  TypePool outofline_param_pool_;
  Set<llvm::StringRef> valid_type_names_;

  // The set of class names declared in this file, used to distinguish class
  // types (produced via their `Make` factory, consumed via their `Checksum`
  // method) from builtin types (produced via a value expression, consumed via
  // a consumer template) in `ProduceValue` and when assigning consumers.
  Set<llvm::StringRef> class_name_set_;
  // Maps each builtin type's spelling to the expression that produces a value
  // of that type, and to the list of templates consuming such a value, for the
  // language being generated. The consumer lists (and `class_consumers_`)
  // reference the caller's `TypeUseParams`, which outlives generation.
  Map<llvm::StringRef, llvm::StringRef> fixed_value_;
  Map<llvm::StringRef, llvm::ArrayRef<llvm::StringRef>> fixed_consumers_;
  llvm::ArrayRef<llvm::StringRef> class_consumers_;
};

// A helper to sum elements of a range.
template <typename T>
static auto Sum(const T& range) -> int {
  return std::accumulate(range.begin(), range.end(), 0);
}

// Given a number of class definitions and the params with which to generate
// them, builds the state that will be used while generating that many classes.
//
// We build the state first and across all the class definitions that will be
// generated so that we can distribute random components across all the
// definitions.
SourceGen::ClassGenState::ClassGenState(SourceGen& gen, int num_classes,
                                        const ClassParams& class_params,
                                        const TypeUseParams& type_use_params,
                                        bool define_decls_out_of_line)
    : generate_bodies_(class_params.inline_function_defs > 0 ||
                       define_decls_out_of_line),
      split_type_pools_(define_decls_out_of_line) {
  public_function_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.public_function_decls, 0,
                          class_params.public_function_decl_params.max_params);
  public_method_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.public_method_decls, 0,
                          class_params.public_method_decl_params.max_params);
  private_function_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.private_function_decls, 0,
                          class_params.private_function_decl_params.max_params);
  private_method_param_counts_ =
      gen.GetShuffledInts(num_classes * class_params.private_method_decls, 0,
                          class_params.private_method_decl_params.max_params);

  // The number of function and method declarations in each class. Every one of
  // these has a return type, and so this is also a guaranteed lower bound on
  // the number of type uses in each class that are allowed to reference the
  // class itself (return types and parameters, as opposed to fields).
  int decls_per_class =
      class_params.public_function_decls + class_params.public_method_decls +
      class_params.private_function_decls + class_params.private_method_decls;
  int num_inline_functions = num_classes * class_params.inline_function_defs;
  method_function_names_ = gen.GetShuffledIdentifiers(
      num_classes * decls_per_class, /*min_length=*/MinMemberNameLength);
  field_names_ =
      gen.GetShuffledIdentifiers(num_classes * class_params.private_field_decls,
                                 /*min_length=*/MinMemberNameLength);
  int num_params =
      Sum(public_function_param_counts_) + Sum(public_method_param_counts_) +
      Sum(private_function_param_counts_) + Sum(private_method_param_counts_);
  // When defining declarations out-of-line, each function and method also gets
  // one guaranteed "mirror" parameter (see `GetOutOfLineParamType`), so reserve
  // a name for each of those as well.
  if (split_type_pools_) {
    num_params += num_classes * decls_per_class;
  }
  param_names_ = gen.GetShuffledIdentifiers(num_params);

  // Build the state for inline function definitions. Their parameter and local
  // counts come from dedicated distributions, and their names from dedicated
  // pools. Inline parameter names use the full length distribution like other
  // parameters (they are emitted once, in the signature, so a varied length
  // stays seed-independent); the inline param popper excludes the class names
  // so a parameter can never shadow a class used by the body's `Make` call.
  // Inline function names are emitted once but are not re-emitted out-of-line
  // like declared functions, so they cannot share the declaration name pool
  // (whose names are emitted a different number of times in the defined-decls
  // pattern); they use a distribution bounded below `MinMemberNameLength` so
  // they stay unique against all declaration, field, and class names by length
  // alone. Local names keep a single fixed length, also below
  // `MinMemberNameLength`: a local is re-emitted a position-dependent number of
  // times by the body's chain of initializers, so a varied length would make
  // the byte total seed-dependent, and the short length keeps locals from
  // colliding with the class names they produce. Locals share their fixed
  // length with part of the parameter length distribution, though, so the
  // local popper excludes the enclosing function's parameter names (see
  // `GenerateInlineFunctionDef`).
  inline_function_param_counts_ =
      gen.GetShuffledInts(num_inline_functions, 0,
                          class_params.inline_function_decl_params.max_params);
  local_counts_ = gen.GetShuffledInts(num_inline_functions, 0,
                                      class_params.max_body_locals);
  int num_inline_params = Sum(inline_function_param_counts_);
  int num_locals = Sum(local_counts_);
  // Inline-definition parameters are consumed by their bodies, giving them a
  // different emission profile than declaration parameters, so they always
  // have their own pool. Each inline definition gets one guaranteed "mirror"
  // parameter (the same mechanism as out-of-line definitions; see
  // `GetInlineParamType`) so that pool has self-absorbing slots and can admit
  // class types. Reserve a name for each.
  num_inline_params += num_inline_functions;
  inline_function_names_ =
      gen.GetShuffledIdentifiers(num_inline_functions, /*min_length=*/2,
                                 /*max_length=*/MinMemberNameLength - 1);
  inline_param_names_ = gen.GetShuffledIdentifiers(num_inline_params);
  local_names_ = gen.GetShuffledIdentifiers(num_locals,
                                            /*min_length=*/LocalNameLength,
                                            /*max_length=*/LocalNameLength);

  // Inline functions add type uses too: one return type plus one per parameter.
  // They also add return-type slots that can reference the enclosing class, so
  // they raise the guaranteed per-class capacity for class references.
  BuildClassAndTypeNames(gen, num_classes, class_params, type_use_params);
}

auto SourceGen::ClassGenState::GetValidTypeUse(TypePool& pool) -> TypeUse {
  // Check that we don't completely wrap the type names by tracking where we
  // started.
  int initial_last_index = pool.last_index;

  // Now search the type uses, starting from the last used index, to find the
  // first valid one.
  for (;;) {
    if (pool.last_index == 0) {
      pool.last_index = pool.uses.size();
    }
    --pool.last_index;
    TypeUse& use = pool.uses[pool.last_index];
    if (valid_type_names_.Contains(use.name)) {
      // Found a valid type use, swap it with the back and pop that off.
      std::swap(pool.uses.back(), use);
      return pool.uses.pop_back_val();
    }

    // `BuildTypePool` caps how many times each class is referenced so that a
    // valid type use always remains here, for any shuffle; this check should
    // never fire.
    CARBON_CHECK(pool.last_index != initial_last_index,
                 "Failed to find a valid type name with {0} candidates, an "
                 "initial index of {1}, and with {2} classes left to emit!",
                 pool.uses.size(), initial_last_index, class_names_.size());
  }
}

// Builds a single pool of `num_types` type-name references, combining declared
// class names with the fixed types from `type_use_params` to roughly match the
// configured weights, and returns it shuffled.
//
// For each of the fixed types, `type_use_params` provides a spelling for both
// Carbon and C++. We distribute our references to declared class names evenly
// to the extent possible. Before all the references are formed, the class names
// are kept in their original unshuffled order; this ensures that any uneven
// sampling of names is done deterministically. At the end the references are
// shuffled to provide an unpredictable order in the generated output.
//
// `max_refs_per_class` caps how many times each class is referenced in this
// pool. This is what guarantees that the `GetValidTypeUse` search can always
// find a valid type and fully drain the pool, for any shuffle.
//
// A reference to a class only becomes a valid type once that class begins being
// defined, and a class's own fields can never reference it (only its return
// types and parameters can). So references to a given class can only land on a
// type use within that class's own non-field declarations or within a later
// class. The tightest case is the last class to be defined: references to it
// can only be placed on its own return types and parameters. Each class is
// guaranteed at least `max_refs_per_class` such slots, regardless of how the
// random parameter counts are distributed, so as long as no class is referenced
// more than that many times every reference can always be placed on some valid
// type use. Any references dropped by this cap are made up with fixed types, so
// the number and spellings of the pooled types -- and thus the byte count --
// are independent of the shuffle.
//
// `valid_type_names_` must already contain the fixed type spellings; the caller
// seeds it once and reuses it across pools.
auto SourceGen::ClassGenState::BuildTypePool(
    SourceGen& gen, int num_types, int max_refs_per_class, bool producible_only,
    bool consumed, const TypeUseParams& type_use_params) -> TypePool {
  TypePool pool;
  if (num_types == 0) {
    return pool;
  }
  pool.uses.reserve(num_types);

  // The spelling and a flag for whether this fixed type is usable in this pool.
  // In a produced pool we only use types that can be constructed directly.
  auto fixed_spelling = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return gen.IsCpp() ? fw.cpp_spelling : fw.carbon_spelling;
  };
  auto fixed_usable = [&](const TypeUseParams::FixedTypeWeight& fw) {
    return !producible_only ||
           !(gen.IsCpp() ? fw.cpp_value : fw.carbon_value).empty();
  };

  // In a consumed pool, assign each appended use a consumer template from its
  // type's list, round-robin per type so the template mix is a deterministic
  // function of the (deterministic) per-type reference counts.
  Map<llvm::StringRef, int> consumer_counters;
  auto append_use = [&](llvm::StringRef name) {
    llvm::StringRef consumer;
    if (consumed) {
      llvm::ArrayRef<llvm::StringRef> consumers =
          class_name_set_.Contains(name)
              ? class_consumers_
              : fixed_consumers_.Lookup(name).value();
      int& counter = consumer_counters.Insert(name, 0).value();
      consumer = consumers[counter++ % consumers.size()];
    }
    pool.uses.push_back({.name = name, .consumer = consumer});
  };

  int type_weight_sum = type_use_params.declared_types_weight;
  for (const auto& fixed_type_weight : type_use_params.fixed_type_weights) {
    if (fixed_usable(fixed_type_weight)) {
      type_weight_sum += fixed_type_weight.weight;
    }
  }

  // Compute the number of declared types used. We expect to have a decent
  // number of repeated names, so we repeatedly append the entire sequence of
  // class names until there is some remainder of names needed.
  int num_classes = class_names_.size();
  int num_declared_types =
      num_types * type_use_params.declared_types_weight / type_weight_sum;
  int full_copies = num_declared_types / num_classes;
  int remainder = num_declared_types % num_classes;
  if (full_copies >= max_refs_per_class) {
    full_copies = max_refs_per_class;
    remainder = 0;
  }

  for ([[maybe_unused]] auto _ : llvm::seq(full_copies)) {
    for (llvm::StringRef name : class_names_) {
      append_use(name);
    }
  }
  // Now append the remainder number of class names. This is where the class
  // names being un-shuffled is essential. We're going to have one extra
  // reference to some fraction of the class names and we want that to be a
  // stable subset.
  for (llvm::StringRef name :
       llvm::ArrayRef(class_names_).slice(0, remainder)) {
    append_use(name);
  }
  num_declared_types = full_copies * num_classes + remainder;
  CARBON_CHECK(static_cast<int>(pool.uses.size()) == num_declared_types);

  // Use each fixed type weight to append the expected number of copies of that
  // type. This isn't exact however, and is designed to stop short.
  for (const auto& fixed_type_weight : type_use_params.fixed_type_weights) {
    if (!fixed_usable(fixed_type_weight)) {
      continue;
    }
    int num_fixed_type = num_types * fixed_type_weight.weight / type_weight_sum;
    for ([[maybe_unused]] auto _ : llvm::seq(num_fixed_type)) {
      append_use(fixed_spelling(fixed_type_weight));
    }
  }

  // If we need a tail of types to hit the exact number, simply round-robin
  // through the usable fixed types without any weighting. With reasonably large
  // numbers of types this won't distort the distribution in an interesting way
  // and is simpler than trying to scale the distribution down.
  while (static_cast<int>(pool.uses.size()) < num_types) {
    for (const auto& fixed_type_weight : type_use_params.fixed_type_weights) {
      if (static_cast<int>(pool.uses.size()) >= num_types) {
        break;
      }
      if (fixed_usable(fixed_type_weight)) {
        append_use(fixed_spelling(fixed_type_weight));
      }
    }
  }
  CARBON_CHECK(static_cast<int>(pool.uses.size()) == num_types);
  pool.last_index = num_types;

  std::shuffle(pool.uses.begin(), pool.uses.end(), gen.rng_);
  return pool;
}

// Builds the class names this file will declare and the type-reference pools
// used throughout those classes, partitioned by emission profile (see the
// type-name accessors). Pools holding produced types (constructed in a body)
// use only directly-producible fixed types. Each pool's cap is the number of
// guaranteed self-absorbing slots per class in that pool (the return slots),
// which is what lets an even distribution always drain.
auto SourceGen::ClassGenState::BuildClassAndTypeNames(
    SourceGen& gen, int num_classes, const ClassParams& class_params,
    const TypeUseParams& type_use_params) -> void {
  // Initially get the sequence of class names without shuffling so we can
  // compute our type name pools from them prior to any shuffling.
  class_names_ =
      gen.GetUniqueIdentifiers(num_classes, /*min_length=*/MinClassNameLength);
  for (llvm::StringRef name : class_names_) {
    class_name_set_.Insert(name);
  }

  // Seed the valid-type set and the value-expression and consumer-template
  // maps with the fixed types. Every fixed type must have a consumer: any of
  // them can be a parameter type, and bodies consume every parameter.
  for (const auto& fw : type_use_params.fixed_type_weights) {
    llvm::StringRef spelling =
        gen.IsCpp() ? fw.cpp_spelling : fw.carbon_spelling;
    valid_type_names_.Insert(spelling);
    fixed_value_.Insert(spelling, gen.IsCpp() ? fw.cpp_value : fw.carbon_value);
    llvm::ArrayRef<llvm::StringRef> consumers =
        gen.IsCpp() ? fw.cpp_consumers : fw.carbon_consumers;
    CARBON_CHECK(!consumers.empty(),
                 "Fixed type `{0}` needs at least one consumer template.",
                 spelling);
    for (llvm::StringRef consumer : consumers) {
      CARBON_CHECK(consumer.contains("{0}"),
                   "Fixed type `{0}` has a consumer template without a name "
                   "placeholder.",
                   spelling);
    }
    fixed_consumers_.Insert(spelling, consumers);
  }
  CARBON_CHECK(!type_use_params.class_consumers.empty(),
               "Class types need at least one consumer template.");
  class_consumers_ = type_use_params.class_consumers;

  int decls_per_class =
      class_params.public_function_decls + class_params.public_method_decls +
      class_params.private_function_decls + class_params.private_method_decls;
  int num_decl_returns = num_classes * decls_per_class;
  int num_decl_params =
      Sum(public_function_param_counts_) + Sum(public_method_param_counts_) +
      Sum(private_function_param_counts_) + Sum(private_method_param_counts_);
  int num_inline_returns = num_classes * class_params.inline_function_defs;
  int num_inline_params = Sum(inline_function_param_counts_);
  int num_fields = num_classes * class_params.private_field_decls;
  // Field types are produced uses only when bodies are generated; otherwise
  // they share the decl pool (see `GetFieldType`).
  int num_produced_fields = generate_bodies_ ? num_fields : 0;

  // Produced pool (one signature, produced): inline-definition return types
  // and, when bodies are generated, field types. Inline returns are the
  // self-absorbers, so the cap is the number of inline definitions per class.
  produced_type_pool_ = BuildTypePool(
      gen, num_inline_returns + num_produced_fields,
      class_params.inline_function_defs,
      /*producible_only=*/true, /*consumed=*/false, type_use_params);

  // Inline-definition parameters are consumed by their bodies, a different
  // emission profile than declaration parameters (consumption adds a
  // per-entry template cost), so they always get their own pool. Each inline
  // definition emits one guaranteed "mirror" parameter beyond its random
  // count, giving `inline_function_defs` self-absorbing slots per class and
  // letting these parameters admit class types.
  int num_inline_mirror_params = num_inline_returns;
  inline_param_pool_ = BuildTypePool(
      gen, num_inline_params + num_inline_mirror_params,
      class_params.inline_function_defs,
      /*producible_only=*/false, /*consumed=*/true, type_use_params);

  if (!split_type_pools_) {
    // Declaration-only functions (no out-of-line definitions): their returns
    // and parameters all appear in exactly one signature and are neither
    // produced nor consumed, as are field types when no bodies are generated.
    // The declaration returns (one per function, guaranteed) are the
    // self-absorbers.
    decl_type_pool_ = BuildTypePool(
        gen,
        num_decl_returns + num_decl_params + (num_fields - num_produced_fields),
        decls_per_class,
        /*producible_only=*/false, /*consumed=*/false, type_use_params);
  } else {
    // Out-of-line definitions: declaration returns appear in three emissions
    // (both signatures plus the `Make` construction in the body); they get
    // their own pool with the declaration returns as self-absorbers. The
    // declaration parameters appear in two signatures and are consumed by the
    // out-of-line body, and get their own pool: each out-of-line-defined
    // function emits one guaranteed "mirror" parameter beyond its random
    // count, so every class has at least `decls_per_class` parameter slots to
    // absorb self-references and the pool can admit class types (cap
    // `decls_per_class`).
    int num_mirror_params = num_classes * decls_per_class;
    outofline_return_pool_ = BuildTypePool(
        gen, num_decl_returns, decls_per_class,
        /*producible_only=*/true, /*consumed=*/false, type_use_params);
    outofline_param_pool_ = BuildTypePool(
        gen, num_decl_params + num_mirror_params, decls_per_class,
        /*producible_only=*/false, /*consumed=*/true, type_use_params);
  }

  std::shuffle(class_names_.begin(), class_names_.end(), gen.rng_);
}

// Some heuristic numbers used when formatting generated code. These heuristics
// are loosely based on what we expect to make Carbon code readable, and might
// not fit as well in C++, but we use the same heuristics across languages for
// simplicity and to make the output in different languages more directly
// comparable.
static constexpr int NumSingleLineFunctionParams = 3;
static constexpr int NumSingleLineMethodParams = 2;
static constexpr int MaxParamsPerLine = 4;

// `extra_params` models any guaranteed parameters emitted beyond the random
// count -- the "mirror" parameter that defined functions gain -- shifting the
// modeled distribution to [extra_params, max + extra_params].
static auto EstimateAvgFunctionDeclLines(SourceGen::FunctionDeclParams params,
                                         int extra_params = 0) -> double {
  // Currently model a uniform distribution [0, max] random parameters. Assume
  // a line break before the first parameter for >3 and after every 4th.
  int param_lines = 0;
  for (int num_params :
       llvm::seq_inclusive(extra_params, params.max_params + extra_params)) {
    if (num_params > NumSingleLineFunctionParams) {
      param_lines += (num_params + MaxParamsPerLine - 1) / MaxParamsPerLine;
    }
  }
  return 1.0 + static_cast<double>(param_lines) / (params.max_params + 1);
}

// See `EstimateAvgFunctionDeclLines` for the meaning of `extra_params`.
static auto EstimateAvgMethodDeclLines(SourceGen::MethodDeclParams params,
                                       int extra_params = 0) -> double {
  // Currently model a uniform distribution [0, max] random parameters. Assume
  // a line break before the first parameter for >2 and after every 4th slot,
  // where for a Carbon method the leading `self` occupies the first slot and
  // so shifts every parameter's slot by one. This models the Carbon emission
  // exactly; a C++ method has no `self` and wraps slightly less, which stays
  // within the looser C++ line tolerance (see `source_gen_test`).
  int param_lines = 0;
  for (int num_params :
       llvm::seq_inclusive(extra_params, params.max_params + extra_params)) {
    if (num_params > NumSingleLineMethodParams) {
      param_lines += 1 + num_params / MaxParamsPerLine;
    }
  }
  return 1.0 + static_cast<double>(param_lines) / (params.max_params + 1);
}

// Estimates the average number of lines in an inline function definition,
// including its signature and body but not the leading comment. The body has,
// on average: an accumulator line, one consumption line per parameter
// (including the guaranteed mirror parameter, modeled by `extra_params` = 1 in
// the signature estimate too; see `EstimateAvgFunctionDeclLines`), half of
// `max_body_locals` local-variable lines (one per local), a write-back line
// whenever there is at least one local, a return line, and a closing brace
// line.
static auto EstimateAvgInlineFunctionDefLines(SourceGen::ClassParams params)
    -> double {
  constexpr int MirrorParams = 1;
  double avg_params =
      params.inline_function_decl_params.max_params / 2.0 + MirrorParams;
  double max_locals = params.max_body_locals;
  double avg_locals = max_locals / 2.0;
  double prob_any_local = max_locals / (max_locals + 1.0);
  return EstimateAvgFunctionDeclLines(params.inline_function_decl_params,
                                      MirrorParams) +
         1.0 + avg_params + avg_locals + prob_any_local + 2.0;
}

// Note that this should match the heuristics used when formatting.
// TODO: See top-level TODO about line estimates and formatting.
static auto EstimateAvgClassDefLines(SourceGen::ClassParams params,
                                     bool define_decls_out_of_line) -> double {
  // Comment line, and class open line.
  double avg = 2.0;

  // When declarations are defined out-of-line, each one gains a guaranteed
  // "mirror" parameter beyond its random count; model that in the signature
  // line estimates.
  int decl_extra_params = define_decls_out_of_line ? 1 : 0;

  // One comment line and blank line per function, plus the function lines.
  avg += (2.0 + EstimateAvgFunctionDeclLines(params.public_function_decl_params,
                                             decl_extra_params)) *
         params.public_function_decls;
  avg += (2.0 + EstimateAvgMethodDeclLines(params.public_method_decl_params,
                                           decl_extra_params)) *
         params.public_method_decls;
  avg += (2.0 + EstimateAvgFunctionDeclLines(
                    params.private_function_decl_params, decl_extra_params)) *
         params.private_function_decls;
  avg += (2.0 + EstimateAvgMethodDeclLines(params.private_method_decl_params,
                                           decl_extra_params)) *
         params.private_method_decls;
  avg += (2.0 + EstimateAvgInlineFunctionDefLines(params)) *
         params.inline_function_defs;

  bool generate_bodies =
      params.inline_function_defs > 0 || define_decls_out_of_line;

  // A blank line and all the fields (if any), including the guaranteed `tag`
  // field when bodies are generated.
  double num_fields =
      params.private_field_decls + (generate_bodies ? 1.0 : 0.0);
  if (num_fields > 0) {
    avg += 1.0 + num_fields;
  }

  // When bodies are generated, each class also gets a nested `Make` factory
  // and a `Checksum` method: for each, a blank separator line, a comment line,
  // a signature line, a return line, and a closing brace line.
  if (generate_bodies) {
    avg += 10.0;
  }

  // Each declared function and method additionally gets an out-of-line
  // definition after the class: a blank separator line, a comment line, a
  // single-line signature, an accumulator line, a consumption line per
  // parameter (the average random count plus the guaranteed mirror parameter,
  // plus one for `self` on methods), a return line, and a closing brace line.
  if (define_decls_out_of_line) {
    auto out_of_line_lines = [](int max_params, bool is_method) {
      return 7.0 + max_params / 2.0 + (is_method ? 1.0 : 0.0);
    };
    avg += out_of_line_lines(params.public_function_decl_params.max_params,
                             /*is_method=*/false) *
           params.public_function_decls;
    avg += out_of_line_lines(params.public_method_decl_params.max_params,
                             /*is_method=*/true) *
           params.public_method_decls;
    avg += out_of_line_lines(params.private_function_decl_params.max_params,
                             /*is_method=*/false) *
           params.private_function_decls;
    avg += out_of_line_lines(params.private_method_decl_params.max_params,
                             /*is_method=*/true) *
           params.private_method_decls;
  }

  // No need to account for the class close line, we have an extra blank line
  // count for the last of the above.
  return avg;
}

auto SourceGen::GenApiFileDenseDecls(int target_lines,
                                     const DenseDeclParams& params)
    -> std::string {
  RawStringOstream source;

  // Figure out how many classes fit in our target lines, each separated by a
  // blank line. We need to account the comment lines below to start the file.
  // Note that we want a blank line after our file comment block, so every class
  // needs a blank line.
  constexpr int NumFileCommentLines = 4;
  double avg_class_lines = EstimateAvgClassDefLines(
      params.class_params, params.define_decls_out_of_line);
  CARBON_CHECK(target_lines > NumFileCommentLines + avg_class_lines,
               "Not enough target lines to generate a single class!");
  // Round to the nearest whole class: truncating can leave the file up to a
  // whole class short of the target, which is a significant fraction of it
  // when classes are large (body-generating classes run to hundreds of
  // lines).
  int num_classes =
      std::lround((target_lines - NumFileCommentLines) / (avg_class_lines + 1));
  int expected_lines =
      NumFileCommentLines + num_classes * (avg_class_lines + 1);

  source << "// Generated " << (!IsCpp() ? "Carbon" : "C++")
         << " source file.\n";
  source << llvm::formatv(
                "// {0} target lines: {1} classes, {2} expected lines",
                target_lines, num_classes, expected_lines)
         << "\n";
  source << "//\n// Generating as an API file with dense declarations.\n";

  // Carbon uses an implicitly imported prelude to get builtin types, but C++
  // requires header files so include those.
  if (IsCpp()) {
    source << "\n";
    // Header for specific integer types like `std::int64_t`.
    source << "#include <cstdint>\n";
    // Header for `std::pair`.
    source << "#include <utility>\n";
  }

  auto class_gen_state =
      ClassGenState(*this, num_classes, params.class_params,
                    params.type_use_params, params.define_decls_out_of_line);
  for ([[maybe_unused]] auto _ : llvm::seq(num_classes)) {
    source << "\n";
    GenerateClassDef(params.class_params, class_gen_state, source);
  }

  // Make sure we consumed all the state.
  CARBON_CHECK(class_gen_state.public_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.public_method_param_counts().empty());
  CARBON_CHECK(class_gen_state.private_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.private_method_param_counts().empty());
  CARBON_CHECK(class_gen_state.class_names().empty());
  CARBON_CHECK(class_gen_state.type_pools_empty());
  // The name pools must also be fully consumed. This is what keeps the byte
  // count stable across seeds: each pool's multiset of identifier lengths is
  // seed-independent, so the pool emits a deterministic total number of bytes
  // only when fully drained.
  CARBON_CHECK(class_gen_state.method_function_names().empty());
  CARBON_CHECK(class_gen_state.field_names().empty());
  CARBON_CHECK(class_gen_state.param_names().empty());
  // Likewise the inline-definition state must be fully consumed.
  CARBON_CHECK(class_gen_state.inline_function_param_counts().empty());
  CARBON_CHECK(class_gen_state.local_counts().empty());
  CARBON_CHECK(class_gen_state.inline_function_names().empty());
  CARBON_CHECK(class_gen_state.inline_param_names().empty());
  CARBON_CHECK(class_gen_state.local_names().empty());

  return source.TakeStr();
}

auto SourceGen::GetShuffledIdentifiers(int number, int min_length,
                                       int max_length, bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> idents =
      GetIdentifiers(number, min_length, max_length, uniform);
  std::shuffle(idents.begin(), idents.end(), rng_);
  return idents;
}

auto SourceGen::GetShuffledUniqueIdentifiers(int number, int min_length,
                                             int max_length, bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4,
               "Cannot trivially guarantee enough distinct, unique identifiers "
               "for lengths <= 3");
  llvm::SmallVector<llvm::StringRef> idents =
      GetUniqueIdentifiers(number, min_length, max_length, uniform);
  std::shuffle(idents.begin(), idents.end(), rng_);
  return idents;
}

auto SourceGen::GetIdentifiers(int number, int min_length, int max_length,
                               bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef> idents = GetIdentifiersImpl(
      number, min_length, max_length, uniform,
      [this](int length, int length_count,
             llvm::SmallVectorImpl<llvm::StringRef>& dest) {
        llvm::append_range(dest,
                           GetSingleLengthIdentifiers(length, length_count));
      });

  return idents;
}

auto SourceGen::GetUniqueIdentifiers(int number, int min_length, int max_length,
                                     bool uniform)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length >= 4,
               "Cannot trivially guarantee enough distinct, unique identifiers "
               "for lengths <= 3");
  llvm::SmallVector<llvm::StringRef> idents =
      GetIdentifiersImpl(number, min_length, max_length, uniform,
                         [this](int length, int length_count,
                                llvm::SmallVectorImpl<llvm::StringRef>& dest) {
                           AppendUniqueIdentifiers(length, length_count, dest);
                         });

  return idents;
}

auto SourceGen::GetSingleLengthIdentifiers(int length, int number)
    -> llvm::ArrayRef<llvm::StringRef> {
  llvm::SmallVector<llvm::StringRef>& idents =
      identifiers_by_length_.Insert(length, {}).value();

  if (static_cast<int>(idents.size()) < number) {
    idents.reserve(number);
    for ([[maybe_unused]] auto _ : llvm::seq<int>(idents.size(), number)) {
      auto ident_storage =
          llvm::MutableArrayRef(reinterpret_cast<char*>(storage_.Allocate(
                                    /*Size=*/length, /*Alignment=*/1)),
                                length);
      GenerateRandomIdentifier(ident_storage);
      llvm::StringRef new_id(ident_storage.data(), length);
      idents.push_back(new_id);
    }
    CARBON_CHECK(static_cast<int>(idents.size()) == number);
  }
  return llvm::ArrayRef(idents).slice(0, number);
}

static auto IdentifierStartChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::SmallVector<char> chars;
    for (char c : llvm::seq_inclusive('A', 'Z')) {
      chars.push_back(c);
    }
    for (char c : llvm::seq_inclusive('a', 'z')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

static auto IdentifierChars() -> llvm::ArrayRef<char> {
  static llvm::SmallVector<char> chars = [] {
    llvm::ArrayRef<char> start_chars = IdentifierStartChars();
    llvm::SmallVector<char> chars(start_chars.begin(), start_chars.end());
    chars.push_back('_');
    for (char c : llvm::seq_inclusive('0', '9')) {
      chars.push_back(c);
    }
    return chars;
  }();
  return chars;
}

static constexpr llvm::StringRef NonCarbonCppKeywords[] = {
    "asm",      "catch",  "do",  "double", "float", "int",  "long",     "new",
    "operator", "signed", "std", "this",   "throw", "try",  "typename", "unix",
    "unsigned", "using",  "xor", "M_E",    "M_El",  "M_PI", "NAN",      "NULL",
};

// Names the generator itself emits with a fixed meaning. A randomly generated
// identifier matching one of these could collide with the generated construct
// (for example, an inline function named `Make` would clash with the class's
// `Make` factory), so they are excluded from identifier generation.
static constexpr llvm::StringRef ReservedGeneratedNames[] = {"Make", "Checksum",
                                                             "acc", "tag"};

// Returns a random identifier string of the specified length.
//
// Ensures this is a valid identifier, avoiding any overlapping syntaxes or
// keywords both in Carbon and C++.
//
// This routine is somewhat expensive and so is useful to cache and reduce the
// frequency of calls. However, each time it is called it computes a completely
// new random identifier and so can be useful to eventually find a distinct
// identifier when needed.
auto SourceGen::GenerateRandomIdentifier(
    llvm::MutableArrayRef<char> dest_storage) -> void {
  llvm::ArrayRef<char> start_chars = IdentifierStartChars();
  llvm::ArrayRef<char> chars = IdentifierChars();

  llvm::StringRef ident(dest_storage.data(), dest_storage.size());
  do {
    dest_storage[0] =
        start_chars[absl::Uniform<int>(rng_, 0, start_chars.size())];
    for (int i : llvm::seq<int>(1, dest_storage.size())) {
      dest_storage[i] = chars[absl::Uniform<int>(rng_, 0, chars.size())];
    }
  } while (
      // TODO: Clean up and simplify this code. With some small refactorings and
      // post-processing we should be able to make this both easier to read and
      // less inefficient.
      llvm::any_of(
          Lex::TokenKind::KeywordTokens,
          [ident](auto token) { return ident == token.fixed_spelling(); }) ||
      llvm::is_contained(NonCarbonCppKeywords, ident) ||
      llvm::is_contained(ReservedGeneratedNames, ident) ||
      ident.ends_with("_t") || ident.ends_with("_MIN") ||
      ident.ends_with("_MAX") || ident.ends_with("_C") ||
      (llvm::is_contained({'i', 'u', 'f'}, ident[0]) &&
       llvm::all_of(ident.substr(1),
                    [](const char c) { return llvm::isDigit(c); })));
}

// Appends a number of unique, random identifiers with a particular length to
// the provided destination vector.
//
// Uses, and when necessary grows, a cached sequence of random identifiers with
// the specified length. Because these are cached, this is efficient to call
// repeatedly, but will not produce a different sequence of identifiers.
auto SourceGen::AppendUniqueIdentifiers(
    int length, int number, llvm::SmallVectorImpl<llvm::StringRef>& dest)
    -> void {
  auto& [count, unique_idents] =
      unique_identifiers_by_length_.Insert(length, {}).value();

  // See if we need to grow our pool of unique identifiers with the requested
  // length.
  if (count < number) {
    // We'll need to insert exactly the requested new unique identifiers. All
    // our other inserts will find an existing entry.
    unique_idents.GrowForInsertCount(count - number);

    // Generate the needed number of identifiers.
    for ([[maybe_unused]] auto _ : llvm::seq<int>(count, number)) {
      // Allocate stable storage for the identifier so we can form stable
      // `StringRef`s to it.
      auto ident_storage =
          llvm::MutableArrayRef(reinterpret_cast<char*>(storage_.Allocate(
                                    /*Size=*/length, /*Alignment=*/1)),
                                length);
      // Repeatedly generate novel identifiers of this length until we find a
      // new unique one.
      for (;;) {
        GenerateRandomIdentifier(ident_storage);
        auto result =
            unique_idents.Insert(llvm::StringRef(ident_storage.data(), length));
        if (result.is_inserted()) {
          break;
        }
      }
    }
    count = number;
  }
  // Append all the identifiers directly out of the set. We make no guarantees
  // about the relative order so we just use the non-deterministic order of the
  // set and avoid additional storage.
  //
  // TODO: It's awkward the `ForEach` here can't early-exit. This just walks the
  // whole set which is harmless if inefficient. We should add early exiting
  // the loop support to `Set` and update this code.
  unique_idents.ForEach([&](llvm::StringRef ident) {
    if (number > 0) {
      dest.push_back(ident);
      --number;
    }
  });
  CARBON_CHECK(number == 0);
}

// An array of the counts that should be used for each identifier length to
// produce our desired distribution.
//
// Note that the zero-based index corresponds to a 1-based length, so the count
// for identifiers of length 1 is at index 0.
static constexpr std::array<int, 64> IdentifierLengthCounts = [] {
  std::array<int, 64> ident_length_counts;
  // For non-uniform distribution, we simulate a distribution roughly based on
  // the observed histogram of identifier lengths, but smoothed a bit and
  // reduced to small counts so that we cycle through all the lengths
  // reasonably quickly. We want sampling of even 10% of NumTokens from this
  // in a round-robin form to not be skewed overly much. This still inherently
  // compresses the long tail as we'd rather have coverage even though it
  // distorts the distribution a bit.
  //
  // The distribution here comes from a script that analyzes source code run
  // over a few directories of LLVM. The script renders a visual ascii-art
  // histogram along with the data for each bucket, and that output is
  // included in comments above each bucket size below to help visualize the
  // rough shape we're aiming for.
  //
  // 1 characters   [3976]  ███████████████████████████████▊
  ident_length_counts[0] = 40;
  // 2 characters   [3724]  █████████████████████████████▊
  ident_length_counts[1] = 40;
  // 3 characters   [4173]  █████████████████████████████████▍
  ident_length_counts[2] = 40;
  // 4 characters   [5000]  ████████████████████████████████████████
  ident_length_counts[3] = 50;
  // 5 characters   [1568]  ████████████▌
  ident_length_counts[4] = 20;
  // 6 characters   [2226]  █████████████████▊
  ident_length_counts[5] = 20;
  // 7 characters   [2380]  ███████████████████
  ident_length_counts[6] = 20;
  // 8 characters   [1786]  ██████████████▎
  ident_length_counts[7] = 18;
  // 9 characters   [1397]  ███████████▏
  ident_length_counts[8] = 12;
  // 10 characters  [ 739]  █████▉
  ident_length_counts[9] = 12;
  // 11 characters  [ 779]  ██████▎
  ident_length_counts[10] = 12;
  // 12 characters  [1344]  ██████████▊
  ident_length_counts[11] = 12;
  // 13 characters  [ 498]  ████
  ident_length_counts[12] = 5;
  // 14 characters  [ 284]  ██▎
  ident_length_counts[13] = 3;
  // 15 characters  [ 172]  █▍
  // 16 characters  [ 278]  ██▎
  // 17 characters  [ 191]  █▌
  // 18 characters  [ 207]  █▋
  for (int i = 14; i < 18; ++i) {
    ident_length_counts[i] = 2;
  }
  // 19 - 63 characters are all <100 but non-zero, and we map them to 1 for
  // coverage despite slightly over weighting the tail.
  for (int i = 18; i < 64; ++i) {
    ident_length_counts[i] = 1;
  }
  return ident_length_counts;
}();

// A template function that implements the common logic of `GetIdentifiers` and
// `GetUniqueIdentifiers`. Most parameters correspond to the parameters of those
// functions. Additionally, an `AppendFunc` callable is provided to implement
// the appending operation.
//
// The main functionality provided here is collecting the correct number of
// identifiers from each of the lengths in the range [min_length, max_length]
// and either in our default representative distribution or a uniform
// distribution.
auto SourceGen::GetIdentifiersImpl(int number, int min_length, int max_length,
                                   bool uniform,
                                   llvm::function_ref<AppendFn> append)
    -> llvm::SmallVector<llvm::StringRef> {
  CARBON_CHECK(min_length <= max_length);
  CARBON_CHECK(
      uniform || max_length <= 64,
      "Cannot produce a meaningful non-uniform distribution of lengths longer "
      "than 64 as those are exceedingly rare in our observed data sets.");

  llvm::SmallVector<llvm::StringRef> idents;
  idents.reserve(number);

  // First, compute the total weight of the distribution so we know how many
  // identifiers we'll get each time we collect from it. For a uniform
  // distribution every length has weight one, so the sum is simply the number
  // of lengths; this also avoids indexing the bounded `IdentifierLengthCounts`
  // table, which only covers lengths up to 64 and which uniform callers are
  // allowed to exceed.
  int num_lengths = max_length - min_length + 1;
  int count_sum = uniform ? num_lengths
                          : Sum(llvm::ArrayRef(IdentifierLengthCounts)
                                    .slice(min_length - 1, num_lengths));
  CARBON_CHECK(count_sum >= 1);

  int number_rem = number % count_sum;

  // Finally, walk through each length in the distribution.
  for (int length : llvm::seq_inclusive(min_length, max_length)) {
    // Scale how many identifiers we want of this length if computing a
    // non-uniform distribution. For uniform, we always take one.
    int scale = uniform ? 1 : IdentifierLengthCounts[length - 1];

    // Now we can compute how many identifiers of this length to request.
    int length_count = (number / count_sum) * scale;
    if (number_rem > 0) {
      int rem_adjustment = std::min(scale, number_rem);
      length_count += rem_adjustment;
      number_rem -= rem_adjustment;
    }
    append(length, length_count, idents);
  }
  CARBON_CHECK(number_rem == 0, "Unexpected number remaining: {0}", number_rem);
  CARBON_CHECK(static_cast<int>(idents.size()) == number,
               "Ended up with {0} identifiers instead of the requested {1}",
               idents.size(), number);

  return idents;
}

// Returns a shuffled sequence of integers in the range [min, max].
//
// The order of the returned integers is random, but each integer in the range
// appears the same number of times in the result, with the number of
// appearances rounded up for lower numbers and rounded down for higher numbers
// in order to exactly produce `number` results.
auto SourceGen::GetShuffledInts(int number, int min, int max)
    -> llvm::SmallVector<int> {
  llvm::SmallVector<int> ints;
  ints.reserve(number);

  // Evenly distribute to each value between min and max.
  int num_values = max - min + 1;
  for (int i : llvm::seq_inclusive(min, max)) {
    int i_count = number / num_values;
    i_count += i < (min + (number % num_values));
    ints.append(i_count, i);
  }
  CARBON_CHECK(static_cast<int>(ints.size()) == number);

  std::shuffle(ints.begin(), ints.end(), rng_);
  return ints;
}

// A helper to pop series of unique identifiers off a sequence of random
// identifiers that may have duplicates.
//
// This is particularly designed to work with the sequences of non-unique
// identifiers produced by `GetShuffledIdentifiers` with the important property
// that while popping off unique identifiers found in the shuffled list, we
// don't change the distribution of identifier lengths.
//
// The uniqueness is only per-instance of the class, and so an instance can be
// used to extract a series of names that share a scope.
//
// It works by scanning the sequence to extract each unique identifier found,
// swapping it to the back and popping it off the list. This does shuffle the
// order, but it isn't expected to do so in an interesting way.
//
// It also provides a fallback path in case there are no unique identifiers left
// which computes fresh, random identifiers with the same length as the next one
// in the sequence until a unique one is found.
//
// For simplicity of the fallback path, the lifetime of the identifiers produced
// is bound to the lifetime of the popper instance, and not the generator as a
// whole. If this is ever a problematic constraint, we can start copying
// fallback identifiers into the generator's storage.
class SourceGen::UniqueIdentifierPopper {
 public:
  // An optional `excluded` set of identifiers is never returned by this
  // popper. The set is referenced, not copied, so a large, long-lived set (for
  // example, all of the file's class names) can be shared across many poppers
  // for free; it must outlive the popper and not change while the popper is in
  // use.
  explicit UniqueIdentifierPopper(
      SourceGen& gen, llvm::SmallVectorImpl<llvm::StringRef>& data,
      const Set<llvm::StringRef>* excluded = nullptr)
      : gen_(&gen), data_(&data), it_(data_->rbegin()), excluded_(excluded) {}

  // The set of identifiers this popper has returned so far. Useful for
  // reserving them in another popper to keep names unique across pools that
  // share a scope.
  auto used() const -> const Set<llvm::StringRef>& { return set_; }

  // Reserves a set of identifiers so this popper will never return them. Used
  // to keep names unique across two pools that share a scope (for example, a
  // class's field names must not collide with its function and method names).
  // The names are copied; prefer the constructor's `excluded` set for large
  // ones.
  auto Reserve(const Set<llvm::StringRef>& names) -> void {
    names.ForEach([&](llvm::StringRef name) { set_.Insert(name); });
  }

  // Pop the next unique identifier that can be found in the data, or synthesize
  // one with a valid length. Always consumes exactly one identifier from the
  // data.
  //
  // Note that the lifetime of the underlying identifier is that of the popper
  // and not the underlying data.
  auto Pop() -> llvm::StringRef {
    for (auto end = data_->rend(); it_ != end; ++it_) {
      if (excluded_ && excluded_->Contains(*it_)) {
        continue;
      }
      auto insert = set_.Insert(*it_);
      if (!insert.is_inserted()) {
        continue;
      }

      if (it_ != data_->rbegin()) {
        std::swap(*data_->rbegin(), *it_);
      }
      CARBON_CHECK(insert.key() == data_->back());
      return data_->pop_back_val();
    }

    // Out of unique elements. Overwrite the back, preserving its length,
    // generating a new identifiers until we find a unique one and return that.
    // This ensures we continue to consume the structure and produce the same
    // size identifiers even in the fallback.
    int length = data_->pop_back_val().size();
    auto fallback_ident_storage =
        llvm::MutableArrayRef(reinterpret_cast<char*>(gen_->storage_.Allocate(
                                  /*Size=*/length, /*Alignment=*/1)),
                              length);
    for (;;) {
      gen_->GenerateRandomIdentifier(fallback_ident_storage);
      auto fallback_id = llvm::StringRef(fallback_ident_storage.data(), length);
      if (excluded_ && excluded_->Contains(fallback_id)) {
        continue;
      }
      if (set_.Insert(fallback_id).is_inserted()) {
        return fallback_id;
      }
    }
  }

 private:
  SourceGen* gen_;
  llvm::SmallVectorImpl<llvm::StringRef>* data_;
  llvm::SmallVectorImpl<llvm::StringRef>::reverse_iterator it_;
  const Set<llvm::StringRef>* excluded_;
  Set<llvm::StringRef> set_;
};

// Generates a function declaration and writes it to the provided stream.
//
// The declaration can be configured with a function name, private modifier,
// whether it is a method, the parameter count, an how indented it is.
//
// This is also provided a collection of identifiers to consume as parameter
// names -- it will use a unique popper to extract unique parameter names from
// this collection.
auto SourceGen::GenerateFunctionDecl(ClassGenState& state, llvm::StringRef name,
                                     bool is_private, bool is_method,
                                     int param_count, llvm::StringRef indent,
                                     llvm::raw_ostream& os,
                                     FunctionSig* captured) -> void {
  // When this declaration is also defined out-of-line (`captured` set), its
  // types come from the twice-emitted out-of-line pools; otherwise from the
  // once-emitted decl pool.
  bool out_of_line = captured != nullptr;
  // Out-of-line-defined functions get one guaranteed extra "mirror" parameter.
  // It ensures every class has at least `decls_per_class` parameter slots that
  // become valid once the class itself does, which is what lets the out-of-line
  // parameter pool admit (self-referential) class types -- see
  // `GetOutOfLineParamType`.
  if (out_of_line) {
    param_count += 1;
  }
  auto param_type = [&] {
    return out_of_line ? state.GetOutOfLineParamType() : state.GetDeclType();
  };
  auto return_type = [&] {
    return out_of_line ? state.GetOutOfLineReturnType() : state.GetDeclType();
  };
  if (captured) {
    captured->name = name;
    captured->is_method = is_method;
    captured->param_names.reserve(param_count);
    captured->param_types.reserve(param_count);
  }
  os << indent << "// TODO: make better comment text\n";
  if (!IsCpp()) {
    os << indent << (is_private ? "private " : "") << "fn " << name;
  } else {
    os << indent;
    if (!is_method) {
      os << "static ";
    }
    os << "auto " << name;
  }

  os << "(";

  if (param_count >
      (is_method ? NumSingleLineMethodParams : NumSingleLineFunctionParams)) {
    os << "\n" << indent << "    ";
  }
  // For Carbon methods, `self` is the first explicit parameter. Its type is
  // omitted, which defaults it to `Self`.
  bool is_carbon_method = is_method && !IsCpp();
  if (is_carbon_method) {
    os << "self";
  }
  // Exclude class names: a declaration defined out-of-line has its parameters
  // in scope of a body that references the return type class via `Make`.
  UniqueIdentifierPopper unique_param_names(*this, state.param_names(),
                                            &state.class_name_set());
  for (int i : llvm::seq(param_count)) {
    // `self` occupies the first slot for Carbon methods, so shift the index
    // used for separators and line wrapping.
    int slot = i + (is_carbon_method ? 1 : 0);
    if (slot > 0) {
      if ((slot % MaxParamsPerLine) == 0) {
        os << ",\n" << indent << "    ";
      } else {
        os << ", ";
      }
    }
    llvm::StringRef param = unique_param_names.Pop();
    ClassGenState::TypeUse type = param_type();
    if (captured) {
      captured->param_names.push_back(param);
      captured->param_types.push_back(type.name);
      captured->param_consumers.push_back(type.consumer);
    }
    if (!IsCpp()) {
      os << param << ": " << type.name;
    } else {
      os << type.name << " " << param;
    }
  }
  os << ")";

  llvm::StringRef ret = return_type().name;
  if (captured) {
    captured->return_type = ret;
  }
  os << " -> " << ret;
  os << ";\n";
}

// Emits an expression of type `i32` (`int` in C++) consuming (reading) the
// value named `name`, by substituting the name for the consumer template's
// `{0}` placeholder. Bodies use this to consume every parameter so that none
// are unused. The per-consumption byte cost depends only on the type pool
// entry (which carries the template) and the name, which keeps totals
// seed-independent.
static auto EmitConsumer(llvm::StringRef consumer, llvm::StringRef name,
                         llvm::raw_ostream& os) -> void {
  CARBON_CHECK(!consumer.empty());
  auto [prefix, suffix] = consumer.split("{0}");
  os << prefix << name << suffix;
}

// Generates an out-of-line definition matching a previously-declared function
// or method, writing it to the provided stream.
//
// The body accumulates a consumption of every parameter (and of `self` for
// methods, via the class's `Checksum`) into a local accumulator, then returns
// a produced value of the return type (a `Make` call for a class type, a
// literal for a builtin). Consuming every binding keeps the generated code
// free of unused-binding warnings, which would otherwise flood benchmark
// output and distort the check benchmarks into measuring diagnostic emission.
//
// The full signature is re-emitted from the captured `sig`, on a single line
// regardless of parameter count -- a simplification relative to the wrapped
// in-class declarations that the line estimates model as such -- and each
// consumption line's cost depends only on the parameter's type pool entry and
// name; because every declared function and method is defined this way, the
// out-of-line type pools are re-emitted (and consumed) in their entirety and
// the byte total stays seed-independent. The accumulator is unconditional:
// there is always at least the mirror parameter to consume.
auto SourceGen::GenerateOutOfLineDef(ClassGenState& state,
                                     llvm::StringRef class_name,
                                     const FunctionSig& sig,
                                     llvm::raw_ostream& os) -> void {
  os << "// TODO: make better comment text\n";
  if (!IsCpp()) {
    os << "fn " << class_name << "." << sig.name;
  } else {
    os << "auto " << class_name << "::" << sig.name;
  }

  os << "(";
  // For Carbon methods, `self` is the first explicit parameter, matching the
  // declaration. Its type is omitted, which defaults it to `Self`.
  bool is_carbon_method = sig.is_method && !IsCpp();
  if (is_carbon_method) {
    os << "self";
  }
  bool first_param = !is_carbon_method;
  for (auto [param, type] : llvm::zip(sig.param_names, sig.param_types)) {
    if (!first_param) {
      os << ", ";
    }
    first_param = false;
    if (!IsCpp()) {
      os << param << ": " << type;
    } else {
      os << type << " " << param;
    }
  }
  os << ") -> " << sig.return_type << " {\n";

  os << (IsCpp() ? "  int acc = 0;\n" : "  var acc: i32 = 0;\n");
  // Methods consume `self` through the class's own `Checksum`; C++ spells the
  // same call through the implicit object parameter.
  if (sig.is_method) {
    os << "  acc = acc + " << (IsCpp() ? "Checksum()" : "self.Checksum()")
       << ";\n";
  }
  for (auto [param, consumer] :
       llvm::zip(sig.param_names, sig.param_consumers)) {
    os << "  acc = acc + ";
    EmitConsumer(consumer, param, os);
    os << ";\n";
  }

  os << "  return ";
  state.ProduceValue(sig.return_type, IsCpp(), os);
  os << ";\n}\n";
}

// Generates an inline function definition (a function with a body) and writes
// it to the provided stream.
//
// The body consumes every parameter into an accumulator (see
// `GenerateOutOfLineDef` for why bodies consume all of their bindings), then
// emits a sequence of local variables -- each initialized by a small
// sub-expression over the previous one -- followed by a `return` that produces
// a value of the return type via `ProduceValue` (a `Make` call for a class
// type, a literal for a builtin). The accumulator and locals all have type
// `i32` (`int` in C++), a copyable builtin, so they type-check regardless of
// the signature.
//
// Determinism: the parameter and local counts come from dedicated
// distributions and the names from dedicated pools; each consumption line's
// cost depends only on the parameter's type pool entry and name; the return
// type comes from the produced pool. The accumulator is unconditional: there
// is always at least the mirror parameter to consume.
auto SourceGen::GenerateInlineFunctionDef(ClassGenState& state,
                                          llvm::StringRef name, int param_count,
                                          int local_count,
                                          llvm::StringRef indent,
                                          llvm::raw_ostream& os) -> void {
  os << indent << "// TODO: make better comment text\n";
  os << indent << (IsCpp() ? "static auto " : "fn ") << name << "(";

  // Every inline definition gets one guaranteed "mirror" parameter beyond the
  // random count: its parameters are consumed and so form their own pool, and
  // the mirror provides that pool's guaranteed self-absorbing slot per class
  // (see `GetInlineParamType`).
  param_count += 1;

  if (param_count > NumSingleLineFunctionParams) {
    os << "\n" << indent << "    ";
  }
  // Parameter names use the full length distribution, so they can collide
  // with a class name; exclude the class names so a parameter never shadows
  // the class produced by this body's `Make` call.
  UniqueIdentifierPopper unique_param_names(*this, state.inline_param_names(),
                                            &state.class_name_set());
  llvm::SmallVector<std::pair<llvm::StringRef, ClassGenState::TypeUse>>
      sig_params;
  sig_params.reserve(param_count);
  for (int i : llvm::seq(param_count)) {
    if (i > 0) {
      if ((i % MaxParamsPerLine) == 0) {
        os << ",\n" << indent << "    ";
      } else {
        os << ", ";
      }
    }
    llvm::StringRef param = unique_param_names.Pop();
    ClassGenState::TypeUse type = state.GetInlineParamType();
    sig_params.push_back({param, type});
    if (!IsCpp()) {
      os << param << ": " << type.name;
    } else {
      os << type.name << " " << param;
    }
  }
  os << ")";
  llvm::StringRef return_type = state.GetProducedType().name;
  os << " -> " << return_type << " {\n";

  std::string body_indent = indent.str() + "  ";

  // Consume every parameter into an accumulator, at a cost keyed to the
  // parameter's type pool entry and name.
  os << body_indent << (IsCpp() ? "int acc = 0;\n" : "var acc: i32 = 0;\n");
  for (auto [param, type] : sig_params) {
    os << body_indent << "acc = acc + ";
    EmitConsumer(type.consumer, param, os);
    os << ";\n";
  }

  // Emit the local variables, each initialized by a small sub-expression. The
  // first uses a literal; each subsequent one references the previous local so
  // that every local except the last is read at least once.
  //
  // The parameter names above use the full length distribution, which overlaps
  // the fixed local-name length -- and identifiers of a given length come from
  // a shared pool, so without exclusion a local could be the *same* identifier
  // as one of this function's parameters. Carbon merely shadows in that case,
  // but C++ rejects redeclaring a parameter in the function's outermost block,
  // so exclude this function's parameter names from the locals. (All the
  // parameters have been popped at this point, so the `used()` set is stable.)
  llvm::SmallVector<llvm::StringRef> locals;
  locals.reserve(local_count);
  UniqueIdentifierPopper unique_local_names(*this, state.local_names(),
                                            &unique_param_names.used());
  for (int i : llvm::seq(local_count)) {
    llvm::StringRef local = unique_local_names.Pop();
    locals.push_back(local);
    if (!IsCpp()) {
      os << body_indent << "var " << local << ": i32 = ";
    } else {
      os << body_indent << "int " << local << " = ";
    }
    if (i == 0) {
      os << "1";
    } else {
      os << locals[i - 1] << " + 1";
    }
    os << ";\n";
  }
  // Consume the last local by assigning it back into the first, so the last
  // local is read too and the body produces no unused-variable warnings. (When
  // there is a single local this is a self-assignment, which still reads it.)
  if (local_count > 0) {
    os << body_indent << locals.front() << " = " << locals.back() << ";\n";
  }

  os << body_indent << "return ";
  state.ProduceValue(return_type, IsCpp(), os);
  os << ";\n";

  os << indent << "}\n";
}

// Generates a class's nested `Make` factory, which returns a value of the class
// constructed from a struct literal. The guaranteed `tag` field comes first,
// then each configured field is produced via `ProduceValue`, recursing through
// earlier classes' `Make` factories for class-typed fields (a DAG that bottoms
// out at builtins, so there is no recursion).
auto SourceGen::GenerateMakeFunction(
    ClassGenState& state, llvm::StringRef class_name,
    llvm::ArrayRef<std::pair<llvm::StringRef, llvm::StringRef>> fields,
    llvm::raw_ostream& os) -> void {
  os << "  // TODO: make better comment text\n";
  os << "  " << (IsCpp() ? "static auto " : "fn ") << "Make() -> " << class_name
     << " {\n";
  os << "    return {";
  llvm::ListSeparator sep;
  os << sep << (IsCpp() ? "0" : ".tag = 0");
  for (auto [field_name, field_type] : fields) {
    os << sep;
    // Carbon uses designated initializers; C++ uses positional aggregate init.
    if (!IsCpp()) {
      os << "." << field_name << " = ";
    }
    state.ProduceValue(field_type, IsCpp(), os);
  }
  os << "};\n  }\n";
}

// Generates a class's `Checksum` method, the consumer counterpart of `Make`:
// the class consumer templates read a value of any class type by calling its
// `Checksum`.
// The body reads the guaranteed `tag` field, which both consumes `self` and
// gives the method a fixed, class-independent cost -- consuming the randomly
// typed fields instead would give field type uses a different emission profile
// than the inline return types they share a pool with.
auto SourceGen::GenerateChecksumFunction(llvm::raw_ostream& os) -> void {
  os << "  // TODO: make better comment text\n";
  if (!IsCpp()) {
    os << "  fn Checksum(self) -> i32 {\n";
    os << "    return self.tag + 1;\n";
  } else {
    os << "  auto Checksum() -> int {\n";
    os << "    return tag + 1;\n";
  }
  os << "  }\n";
}

// Generate a class definition and write it to the provided stream.
//
// The structure of the definition is guided by the `params` provided, and it
// consumes the provided state.
auto SourceGen::GenerateClassDef(const ClassParams& params,
                                 ClassGenState& state, llvm::raw_ostream& os)
    -> void {
  llvm::StringRef name = state.class_names().pop_back_val();
  os << "// TODO: make better comment text\n";
  os << "class " << name << " {\n";
  if (IsCpp()) {
    os << " public:\n";
  }

  // Field types can't be the class we're currently declaring (a class can't
  // contain itself), so collect them before marking this class as a valid
  // type.
  llvm::SmallVector<llvm::StringRef> field_type_names;
  field_type_names.reserve(params.private_field_decls);
  for ([[maybe_unused]] auto _ : llvm::seq(params.private_field_decls)) {
    field_type_names.push_back(state.GetFieldType().name);
  }

  // Mark this class as now a valid type now that field type names have been
  // collected. We can reference this class from functions and methods within
  // the definition.
  state.AddValidTypeName(name);

  // Member names are in scope of every body this class emits (inline
  // definitions, out-of-line definitions, and the `Make` factory), all of
  // which may reference class names via `Make` calls, so exclude the class
  // names from them. Inline function names can't collide with class names by
  // length alone and need no exclusion.
  UniqueIdentifierPopper unique_member_names(
      *this, state.method_function_names(), &state.class_name_set());
  UniqueIdentifierPopper unique_inline_names(*this,
                                             state.inline_function_names());

  // When generating out-of-line definitions, capture each declared function and
  // method signature so a matching definition can be emitted after the class.
  bool define_out_of_line = state.define_decls_out_of_line();
  llvm::SmallVector<FunctionSig> decl_sigs;
  if (define_out_of_line) {
    decl_sigs.reserve(
        params.public_function_decls + params.public_method_decls +
        params.private_function_decls + params.private_method_decls);
  }
  auto capture_sig = [&]() -> FunctionSig* {
    return define_out_of_line ? &decl_sigs.emplace_back() : nullptr;
  };

  llvm::ListSeparator line_sep("\n");
  for ([[maybe_unused]] auto _ : llvm::seq(params.public_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/false,
                         state.public_function_param_counts().pop_back_val(),
                         /*indent=*/"  ", os, capture_sig());
  }
  // Inline function definitions are emitted as public class functions with
  // bodies. They use their own dedicated name pool so their names don't perturb
  // the declaration name pools.
  for ([[maybe_unused]] auto _ : llvm::seq(params.inline_function_defs)) {
    os << line_sep;
    GenerateInlineFunctionDef(
        state, unique_inline_names.Pop(),
        state.inline_function_param_counts().pop_back_val(),
        state.local_counts().pop_back_val(), /*indent=*/"  ", os);
  }
  for ([[maybe_unused]] auto _ : llvm::seq(params.public_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/false,
                         /*is_method=*/true,
                         state.public_method_param_counts().pop_back_val(),
                         /*indent=*/"  ", os, capture_sig());
  }

  if (IsCpp()) {
    os << "\n private:\n";
    // Reset the separator.
    line_sep = llvm::ListSeparator("\n");
  }

  for ([[maybe_unused]] auto _ : llvm::seq(params.private_function_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/false,
                         state.private_function_param_counts().pop_back_val(),
                         /*indent=*/"  ", os, capture_sig());
  }
  for ([[maybe_unused]] auto _ : llvm::seq(params.private_method_decls)) {
    os << line_sep;
    GenerateFunctionDecl(state, unique_member_names.Pop(), /*is_private=*/true,
                         /*is_method=*/true,
                         state.private_method_param_counts().pop_back_val(),
                         /*indent=*/"  ", os, capture_sig());
  }

  // Field names come from a separate pool, but must still be unique within the
  // class, so exclude the function and method names already used here, and
  // class names for the same reason as members. Pair each field name with its
  // type for both the `Make` factory and the field declarations.
  UniqueIdentifierPopper unique_field_names(*this, state.field_names(),
                                            &state.class_name_set());
  unique_field_names.Reserve(unique_member_names.used());
  llvm::SmallVector<std::pair<llvm::StringRef, llvm::StringRef>> fields;
  fields.reserve(field_type_names.size());
  for (llvm::StringRef type_name : field_type_names) {
    fields.push_back({unique_field_names.Pop(), type_name});
  }

  // When this file generates bodies, the `Make` factory and the fields go in a
  // final (re-opened) public section for C++, so that the factory is callable
  // and the fields are public -- which makes the class an aggregate that
  // `Make`'s brace initializer can construct. This fidelity tradeoff is
  // confined to body-generating configurations: in the pure-declaration
  // pattern no `Make` is emitted and the fields stay in the private section.
  // Carbon has no access sections and uses per-declaration `private`.
  bool generate_bodies = state.generates_bodies();
  if (IsCpp() && generate_bodies) {
    os << "\n public:\n";
    line_sep = llvm::ListSeparator("\n");
  }

  // Emit the nested `Make` factory and `Checksum` consumer, but only when this
  // file generates bodies (which are the only callers of them).
  if (generate_bodies) {
    os << line_sep;
    GenerateMakeFunction(state, name, fields, os);
    os << line_sep;
    GenerateChecksumFunction(os);
  }

  os << line_sep;
  // The guaranteed `tag` field backing `Checksum` comes first, matching the
  // positional initialization order in the C++ `Make`.
  if (generate_bodies) {
    os << (IsCpp() ? "  int tag;\n" : "  private var tag: i32;\n");
  }
  for (auto [field_name, type_name] : fields) {
    if (!IsCpp()) {
      os << "  private var " << field_name << ": " << type_name << ";\n";
    } else {
      os << "  " << type_name << " " << field_name << ";\n";
    }
  }
  os << "}" << (IsCpp() ? ";" : "") << "\n";

  // Emit an out-of-line definition for every declared function and method,
  // after the class.
  for (const FunctionSig& sig : decl_sigs) {
    os << "\n";
    GenerateOutOfLineDef(state, name, sig, os);
  }
}

}  // namespace Carbon::Testing
