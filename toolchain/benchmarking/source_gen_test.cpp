// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/benchmarking/source_gen.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "common/raw_string_ostream.h"
#include "common/set.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "testing/base/global_exe_path.h"
#include "toolchain/base/install_paths_test_helpers.h"
#include "toolchain/driver/driver.h"

namespace Carbon::Testing {
namespace {

using ::testing::AllOf;
using ::testing::ContainerEq;
using ::testing::Contains;
using ::testing::Each;
using ::testing::Eq;
using ::testing::Ge;
using ::testing::Gt;
using ::testing::HasSubstr;
using ::testing::Le;
using ::testing::MatchesRegex;
using ::testing::SizeIs;

// Tiny helper to sum the sizes of a range of ranges. Uses a template to avoid
// hard coding any specific types for the two ranges.
template <typename T>
static auto SumSizes(const T& range) -> ssize_t {
  ssize_t sum = 0;
  for (const auto& inner_range : range) {
    sum += inner_range.size();
  }
  return sum;
}

// Counts the number of lines (newline characters) in some source text.
static auto CountLines(llvm::StringRef source) -> ssize_t {
  return llvm::count(source, '\n');
}

TEST(SourceGenTest, Identifiers) {
  SourceGen gen;

  auto idents = gen.GetShuffledIdentifiers(1000);
  EXPECT_THAT(idents.size(), Eq(1000));
  for (llvm::StringRef ident : idents) {
    EXPECT_THAT(ident, MatchesRegex("[A-Za-z][A-Za-z0-9_]*"));
  }

  // We should have at least one identifier of each length [1, 64]. The exact
  // distribution is an implementation detail designed to vaguely match the
  // expected distribution in source code.
  for (int size : llvm::seq_inclusive(1, 64)) {
    EXPECT_THAT(idents, Contains(SizeIs(size)));
  }

  // Check that identifiers 4 characters or shorter are more common than longer
  // lengths. This is a very rough way of double checking that we got the
  // intended distribution.
  for (int short_size : llvm::seq_inclusive(1, 4)) {
    int short_count = llvm::count_if(idents, [&](auto ident) {
      return static_cast<int>(ident.size()) == short_size;
    });
    for (int long_size : llvm::seq_inclusive(5, 64)) {
      EXPECT_THAT(short_count, Gt(llvm::count_if(idents, [&](auto ident) {
                    return static_cast<int>(ident.size()) == long_size;
                  })));
    }
  }

  // Check that repeated calls are different in interesting ways, but have the
  // exact same total bytes.
  ssize_t idents_size_sum = SumSizes(idents);
  for ([[maybe_unused]] auto _ : llvm::seq(10)) {
    auto idents2 = gen.GetShuffledIdentifiers(1000);
    EXPECT_THAT(idents2, SizeIs(1000));
    // Should be (at least) a different shuffle of identifiers.
    EXPECT_THAT(idents2, Not(ContainerEq(idents)));
    // But the sum of lengths should be identical.
    EXPECT_THAT(SumSizes(idents2), Eq(idents_size_sum));
  }

  // Check length constraints have the desired effect.
  idents =
      gen.GetShuffledIdentifiers(1000, /*min_length=*/10, /*max_length=*/20);
  EXPECT_THAT(idents, Each(SizeIs(AllOf(Ge(10), Le(20)))));
}

// For fixed parameters, the total number of bytes across the returned
// identifiers must not depend on the random seed, even though the specific
// identifiers do. This checks that across a range of parameters and across many
// freshly-seeded generators (each `SourceGen` gets an independent random seed).
TEST(SourceGenTest, IdentifierByteSumStableAcrossSeeds) {
  struct Config {
    int number;
    int min_length;
    int max_length;
    bool uniform;
    bool unique;
  };
  // A spread of parameters including: the default range, narrow ranges, the
  // single-length extreme, uniform distributions, and a uniform range with a
  // `max_length` well beyond the 64 limit that only the uniform path allows.
  Config configs[] = {
      {.number = 1000, .min_length = 1, .max_length = 64, .uniform = false},
      {.number = 1000, .min_length = 4, .max_length = 64, .uniform = false},
      {.number = 999, .min_length = 1, .max_length = 64, .uniform = false},
      {.number = 1000, .min_length = 10, .max_length = 20, .uniform = false},
      {.number = 1000, .min_length = 8, .max_length = 8, .uniform = false},
      {.number = 100, .min_length = 10, .max_length = 19, .uniform = true},
      {.number = 97, .min_length = 10, .max_length = 19, .uniform = true},
      {.number = 500, .min_length = 50, .max_length = 200, .uniform = true},
      {.number = 1000,
       .min_length = 4,
       .max_length = 64,
       .uniform = false,
       .unique = true},
      {.number = 1000,
       .min_length = 4,
       .max_length = 4,
       .uniform = false,
       .unique = true},
      {.number = 200,
       .min_length = 30,
       .max_length = 120,
       .uniform = true,
       .unique = true},
  };

  for (const Config& c : configs) {
    SCOPED_TRACE(llvm::formatv(
        "Config: number={0} min_length={1} max_length={2} uniform={3} "
        "unique={4}",
        c.number, c.min_length, c.max_length, c.uniform, c.unique));
    std::optional<ssize_t> expected_sum;
    bool any_different = false;
    std::optional<llvm::SmallVector<std::string>> first;
    constexpr int NumSeeds = 8;
    for (int seed : llvm::seq(NumSeeds)) {
      // Each iteration constructs a fresh generator with an independent random
      // seed; the traced index identifies which iteration failed.
      SCOPED_TRACE(llvm::formatv("Seed iteration: {0}", seed));
      SourceGen gen;
      auto idents = c.unique
                        ? gen.GetShuffledUniqueIdentifiers(
                              c.number, c.min_length, c.max_length, c.uniform)
                        : gen.GetShuffledIdentifiers(c.number, c.min_length,
                                                     c.max_length, c.uniform);
      EXPECT_THAT(idents, SizeIs(c.number));
      EXPECT_THAT(idents,
                  Each(SizeIs(AllOf(Ge(c.min_length), Le(c.max_length)))));

      ssize_t sum = SumSizes(idents);
      if (!expected_sum) {
        expected_sum = sum;
        first.emplace(idents.begin(), idents.end());
        continue;
      }
      // The byte sum must be identical regardless of the seed.
      EXPECT_THAT(sum, Eq(*expected_sum));
      if (!llvm::equal(idents, *first)) {
        any_different = true;
      }
    }
    // Sanity check that the generators really are producing different content,
    // so that the invariance check above is meaningful rather than trivially
    // passing on identical output.
    EXPECT_TRUE(any_different);
  }
}

TEST(SourceGenTest, UniformIdentifiers) {
  SourceGen gen;
  // Check that uniform identifier length results in exact coverage of each
  // possible length for an easy case, both without and with a remainder.
  auto idents =
      gen.GetShuffledIdentifiers(100, /*min_length=*/10, /*max_length=*/19,
                                 /*uniform=*/true);
  EXPECT_THAT(idents, Contains(SizeIs(10)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(11)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(12)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(13)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(14)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(15)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(16)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(17)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(18)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(19)).Times(10));

  idents = gen.GetShuffledIdentifiers(97, /*min_length=*/10, /*max_length=*/19,
                                      /*uniform=*/true);
  EXPECT_THAT(idents, Contains(SizeIs(10)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(11)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(12)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(13)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(14)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(15)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(16)).Times(10));
  EXPECT_THAT(idents, Contains(SizeIs(17)).Times(9));
  EXPECT_THAT(idents, Contains(SizeIs(18)).Times(9));
  EXPECT_THAT(idents, Contains(SizeIs(19)).Times(9));
}

// Largely covered by `Identifiers` and `UniformIdentifiers`, but need to check
// for uniqueness specifically.
TEST(SourceGenTest, UniqueIdentifiers) {
  SourceGen gen;

  auto unique = gen.GetShuffledUniqueIdentifiers(1000);
  EXPECT_THAT(unique.size(), Eq(1000));
  Set<llvm::StringRef> set;
  for (llvm::StringRef ident : unique) {
    EXPECT_THAT(ident, MatchesRegex("[A-Za-z][A-Za-z0-9_]*"));
    EXPECT_TRUE(set.Insert(ident).is_inserted())
        << "Colliding identifier: " << ident;
  }

  // Check single length specifically where uniqueness is the most challenging.
  set.Clear();
  unique = gen.GetShuffledUniqueIdentifiers(1000, /*min_length=*/4,
                                            /*max_length=*/4);
  for (llvm::StringRef ident : unique) {
    EXPECT_TRUE(set.Insert(ident).is_inserted())
        << "Colliding identifier: " << ident;
  }
}

// Check that the source code compiles cleanly: no errors, and also no other
// diagnostic output. Generated code must be entirely warning-free -- warnings
// would distort compile benchmarks into measuring diagnostic emission rather
// than compilation, and flood benchmark output.
auto TestCompile(llvm::StringRef source) -> bool {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  InstallPaths installation(
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath()));
  RawStringOstream error_stream;
  Driver driver(fs, &installation, /*input_stream=*/nullptr, &llvm::outs(),
                &error_stream);

  AddPreludeFilesToVfs(installation, fs);

  fs->addFile("test.carbon", /*ModificationTime=*/0,
              llvm::MemoryBuffer::getMemBuffer(source));
  bool success = driver
                     .RunCommand({"compile", "--phase=check",
                                  "--no-include-carbon-core", "test.carbon"})
                     .success;
  std::string errors = error_stream.TakeStr();
  EXPECT_TRUE(errors.empty()) << errors;
  return success && errors.empty();
}

TEST(SourceGenTest, GenApiFileDenseDeclsTest) {
  SourceGen gen;

  std::string source =
      gen.GenApiFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 1% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(950), Le(1050))));

  // Make sure we generated valid Carbon code.
  EXPECT_TRUE(TestCompile(source));
}

TEST(SourceGenTest, GenApiFileDenseDeclsCppTest) {
  SourceGen gen(SourceGen::Language::Cpp);

  // Generate a 1000-line file which is enough to have a reasonably accurate
  // line count estimate and have a few classes.
  std::string source =
      gen.GenApiFileDenseDecls(1000, SourceGen::DenseDeclParams{});
  // Should be within 10% of the requested line count.
  EXPECT_THAT(source, Contains('\n').Times(AllOf(Ge(900), Le(1100))));

  // TODO: When the driver supports compiling C++ code as easily as Carbon, we
  // should test that the generated C++ code is valid.
}

// The central benchmarking invariant: for a fixed language, line target, and
// generation parameters, the generated source must always have the exact same
// number of lines and bytes regardless of the random seed, even though the
// actual content (identifiers, ordering) differs from run to run.
TEST(SourceGenTest, GenApiFileDenseDeclsStableSizeAcrossSeeds) {
  for (SourceGen::Language language :
       {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
    // Line targets ranging from barely enough for one class up to reasonably
    // large files.
    for (int target_lines : {200, 1000, 5000, 20000}) {
      std::optional<size_t> expected_bytes;
      std::optional<ssize_t> expected_lines;
      std::optional<std::string> first_source;
      bool any_different = false;

      constexpr int NumSeeds = 16;
      for (int _ : llvm::seq(NumSeeds)) {
        // A fresh generator gets an independent random seed.
        SourceGen gen(language);
        std::string source = gen.GenApiFileDenseDecls(
            target_lines, SourceGen::DenseDeclParams{});

        if (!expected_bytes) {
          expected_bytes = source.size();
          expected_lines = CountLines(source);
          first_source = source;
          continue;
        }
        EXPECT_THAT(source.size(), Eq(*expected_bytes))
            << "Byte count varied across seeds for language="
            << static_cast<int>(language) << " target_lines=" << target_lines;
        EXPECT_THAT(CountLines(source), Eq(*expected_lines))
            << "Line count varied across seeds for language="
            << static_cast<int>(language) << " target_lines=" << target_lines;
        if (source != *first_source) {
          any_different = true;
        }
      }
      // Sanity check that we really are shuffling content across seeds,
      // otherwise the invariance check above is meaningless.
      EXPECT_TRUE(any_different)
          << "Expected different source across seeds for language="
          << static_cast<int>(language) << " target_lines=" << target_lines;
    }
  }
}

// Like the above, but exercises non-default class parameters to check that the
// stable-size invariant holds for the general machinery and not just the
// default shape of classes. Different parameters produce different sizes, but
// for any fixed set of parameters the size must be seed-independent.
TEST(SourceGenTest, GenApiFileDenseDeclsStableSizeWithVariedParams) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // Function declarations only: no methods and no fields.
  param_set.push_back({.class_params = {.public_function_decls = 20,
                                        .public_method_decls = 0,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 0}});
  // Functions and methods with large parameter counts to exercise the
  // line-wrapping heuristics.
  param_set.push_back(
      {.class_params = {.public_function_decls = 2,
                        .public_function_decl_params = {.max_params = 16},
                        .public_method_decls = 4,
                        .public_method_decl_params = {.max_params = 16},
                        .private_function_decls = 0,
                        .private_method_decls = 0,
                        .private_field_decls = 0}});
  // The default shape scaled up 2x.
  param_set.push_back({.class_params = {.public_function_decls = 8,
                                        .public_method_decls = 20,
                                        .private_function_decls = 4,
                                        .private_method_decls = 16,
                                        .private_field_decls = 12}});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      std::optional<size_t> expected_bytes;
      std::optional<ssize_t> expected_lines;
      constexpr int NumSeeds = 12;
      for (int _ : llvm::seq(NumSeeds)) {
        SourceGen gen(language);
        std::string source = gen.GenApiFileDenseDecls(5000, params);
        if (!expected_bytes) {
          expected_bytes = source.size();
          expected_lines = CountLines(source);
          continue;
        }
        EXPECT_THAT(source.size(), Eq(*expected_bytes));
        EXPECT_THAT(CountLines(source), Eq(*expected_lines));
      }
    }
  }
}

// Stresses the type-name validity constraints with field-heavy classes. Fields
// cannot reference the class currently being defined (nor any not-yet-defined
// class), so with these shapes almost every type use must be a fixed type or an
// earlier class; the generator's per-class reference cap is what keeps
// `GetValidTypeUse` from running out of valid names, for any shuffle. Covers
// many seeds and extreme shapes, and also checks that the byte and line counts
// are seed-independent and that the output compiles.
TEST(SourceGenTest, GenApiFileDenseDeclsRobustForFieldHeavyParams) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // Many fields with only a couple of functions/methods to absorb references.
  param_set.push_back({.class_params = {.public_function_decls = 1,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 30}});
  // A single method with a very large number of fields.
  param_set.push_back({.class_params = {.public_function_decls = 0,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 50}});
  // Fields only: no functions or methods at all, so no class can be referenced
  // as a type and the type name pool is built entirely from fixed types.
  param_set.push_back({.class_params = {.public_function_decls = 0,
                                        .public_method_decls = 0,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 16}});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      std::optional<size_t> expected_bytes;
      std::optional<ssize_t> expected_lines;
      // Exhausting the valid type names is shuffle-dependent, so use many
      // seeds.
      constexpr int NumSeeds = 32;
      for (int _ : llvm::seq(NumSeeds)) {
        SourceGen gen(language);
        std::string source = gen.GenApiFileDenseDecls(3000, params);
        if (!expected_bytes) {
          expected_bytes = source.size();
          expected_lines = CountLines(source);
          // The generated Carbon must be valid: every emitted field type must
          // be a fixed type or a class defined earlier in the file.
          if (language == SourceGen::Language::Carbon) {
            EXPECT_TRUE(TestCompile(source));
          }
          continue;
        }
        EXPECT_THAT(source.size(), Eq(*expected_bytes));
        EXPECT_THAT(CountLines(source), Eq(*expected_lines));
      }
    }
  }
}

// Inline function definitions: some functions are emitted with a body rather
// than as a forward declaration. The bodies must keep the line and byte counts
// seed-independent, must vary their content across seeds, and must compile --
// including bodies for functions that return non-copyable class types, which
// work by constructing the value via the class's `Make` factory.
TEST(SourceGenTest, GenApiFileDenseDeclsInlineBodies) {
  SourceGen::DenseDeclParams params;
  params.class_params.inline_function_defs = 3;
  params.class_params.max_body_locals = 4;

  for (SourceGen::Language language :
       {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
    std::optional<size_t> expected_bytes;
    std::optional<ssize_t> expected_lines;
    std::optional<std::string> first_source;
    bool any_different = false;

    constexpr int NumSeeds = 16;
    for (int _ : llvm::seq(NumSeeds)) {
      SourceGen gen(language);
      std::string source = gen.GenApiFileDenseDecls(2000, params);

      if (!expected_bytes) {
        expected_bytes = source.size();
        expected_lines = CountLines(source);
        first_source = source;
        // The bodies should actually be present and consume their parameters
        // into the accumulator, including class-typed ones via `Checksum`.
        EXPECT_THAT(source, HasSubstr("return "));
        EXPECT_THAT(source, HasSubstr("acc = acc + "));
        EXPECT_THAT(source, HasSubstr(".Checksum()"));
        // The generated Carbon must compile, including the `Make`-producing
        // bodies of functions returning non-copyable class types.
        if (language == SourceGen::Language::Carbon) {
          EXPECT_TRUE(TestCompile(source));
        }
        continue;
      }
      EXPECT_THAT(source.size(), Eq(*expected_bytes))
          << "Byte count varied across seeds for language="
          << static_cast<int>(language);
      EXPECT_THAT(CountLines(source), Eq(*expected_lines))
          << "Line count varied across seeds for language="
          << static_cast<int>(language);
      if (source != *first_source) {
        any_different = true;
      }
    }
    EXPECT_TRUE(any_different);
  }
}

// Inline bodies must also stay robust and deterministic under extreme shapes:
// many small bodies, large bodies, and field-heavy classes (whose functions
// often return non-copyable class types) all at once, across many seeds.
TEST(SourceGenTest, GenApiFileDenseDeclsInlineBodiesRobust) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // Inline-heavy classes with large bodies and few other declarations.
  param_set.push_back({.class_params = {.public_function_decls = 1,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 4,
                                        .inline_function_defs = 8,
                                        .max_body_locals = 12}});
  // Field-heavy classes with a few inline bodies; most functions return
  // class types, exercising the `Make`-production path heavily.
  param_set.push_back({.class_params = {.public_function_decls = 1,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 24,
                                        .inline_function_defs = 2,
                                        .max_body_locals = 3}});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      std::optional<size_t> expected_bytes;
      std::optional<ssize_t> expected_lines;
      constexpr int NumSeeds = 24;
      for (int _ : llvm::seq(NumSeeds)) {
        SourceGen gen(language);
        std::string source = gen.GenApiFileDenseDecls(3000, params);
        if (!expected_bytes) {
          expected_bytes = source.size();
          expected_lines = CountLines(source);
          if (language == SourceGen::Language::Carbon) {
            EXPECT_TRUE(TestCompile(source));
          }
          continue;
        }
        EXPECT_THAT(source.size(), Eq(*expected_bytes));
        EXPECT_THAT(CountLines(source), Eq(*expected_lines));
      }
    }
  }
}

// Scans generated Carbon source and collects, for each function definition,
// its parameter names (spelled `NAME: Type` in the signature) and its body's
// local-variable names (spelled `var NAME: ...`), checking that no local
// collides with a parameter of the same function. Also accumulates all
// parameter and local names seen across the file into the two out-params so
// the caller can check the scan isn't vacuous.
static auto CheckBodyLocalsAvoidParamNames(llvm::StringRef source,
                                           Set<llvm::StringRef>* all_params,
                                           Set<llvm::StringRef>* all_locals)
    -> void {
  Set<llvm::StringRef> func_params;
  bool in_signature = false;
  // Brace depth within a function body; zero when outside a body. Control-flow
  // blocks within a body open nested braces.
  int body_depth = 0;
  llvm::SmallVector<llvm::StringRef> lines;
  source.split(lines, '\n');
  for (llvm::StringRef line : lines) {
    llvm::StringRef trimmed = line.trim();
    if (!in_signature && body_depth == 0 &&
        (trimmed.starts_with("fn ") || trimmed.starts_with("private fn "))) {
      in_signature = true;
      func_params.Clear();
    }
    if (in_signature) {
      // Collect the identifier preceding each `:` on this signature line;
      // within a signature those are exactly the parameter names.
      for (auto [i, c] : llvm::enumerate(line)) {
        if (c != ':') {
          continue;
        }
        size_t begin = i;
        while (begin > 0 &&
               (llvm::isAlnum(line[begin - 1]) || line[begin - 1] == '_')) {
          --begin;
        }
        if (begin == i) {
          continue;
        }
        llvm::StringRef name = line.substr(begin, i - begin);
        func_params.Insert(name);
        all_params->Insert(name);
      }
      if (trimmed.ends_with(";")) {
        // A forward declaration; no body follows.
        in_signature = false;
      } else if (trimmed.ends_with("{")) {
        in_signature = false;
        body_depth = 1;
      }
    } else if (body_depth > 0) {
      if (trimmed.starts_with("var ")) {
        llvm::StringRef name =
            trimmed.drop_front(strlen("var ")).take_until([](char c) {
              return c == ':';
            });
        all_locals->Insert(name);
        EXPECT_FALSE(func_params.Contains(name))
            << "Local `" << name
            << "` collides with a parameter of the same function.";
      } else {
        // Track the brace depth. Note that a line like `} else {` both
        // closes and opens a brace for a net change of zero.
        if (trimmed.starts_with("}")) {
          --body_depth;
        }
        if (trimmed.ends_with("{")) {
          ++body_depth;
        }
      }
    }
  }
}

// Body locals and parameters draw from identifier pools that share strings per
// length, and parameter names span the local-name length, so the generator
// must explicitly keep a body's locals disjoint from its parameters: Carbon
// would just shadow, but the same name pools feed C++ generation, where
// redeclaring a parameter in the function's outermost block is an error that
// would abort compile benchmarks. Scan the generated Carbon (the name pools
// are language-independent) across seeds and both body-generating patterns.
TEST(SourceGenTest, GenApiFileDenseDeclsBodyLocalsAvoidParamNames) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // Inline-heavy classes: enough parameter names that the length distribution
  // reaches the fixed local-name length, plus many locals.
  param_set.push_back({.class_params = {.public_function_decls = 1,
                                        .public_method_decls = 1,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 4,
                                        .inline_function_defs = 8,
                                        .max_body_locals = 12}});
  // The split pattern with the benchmarked shape of inline definitions.
  param_set.push_back(
      {.class_params = {.inline_function_defs = 8, .max_body_locals = 6},
       .define_decls_out_of_line = true});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    Set<llvm::StringRef> all_params;
    Set<llvm::StringRef> all_locals;
    llvm::SmallVector<std::string> sources;
    constexpr int NumSeeds = 8;
    for (int _ : llvm::seq(NumSeeds)) {
      SourceGen gen;
      sources.push_back(gen.GenApiFileDenseDecls(5000, params));
      CheckBodyLocalsAvoidParamNames(sources.back(), &all_params, &all_locals);
    }
    // Check the scan wasn't vacuous: across the file the parameter and local
    // name pools really do share identifiers (they must only stay disjoint
    // within a single function), so the hazard is genuinely exercised.
    bool any_shared = false;
    all_params.ForEach([&](llvm::StringRef name) {
      any_shared = any_shared || all_locals.Contains(name);
    });
    EXPECT_TRUE(any_shared)
        << "Expected parameter and local name pools to share identifiers.";
  }
}

// Out-of-line definitions: every declared function and method is additionally
// defined out-of-line after its class (alongside the in-class inline
// definitions). The line and byte counts must stay seed-independent, the
// content must vary across seeds, and the result must compile -- including the
// out-of-line bodies that produce class-typed returns via `Make`.
TEST(SourceGenTest, GenApiFileDenseDeclsOutOfLineDefs) {
  SourceGen::DenseDeclParams params;
  params.define_decls_out_of_line = true;
  params.class_params.inline_function_defs = 2;
  params.class_params.max_body_locals = 3;

  for (SourceGen::Language language :
       {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
    std::optional<size_t> expected_bytes;
    std::optional<ssize_t> expected_lines;
    std::optional<std::string> first_source;
    bool any_different = false;

    constexpr int NumSeeds = 16;
    for (int _ : llvm::seq(NumSeeds)) {
      SourceGen gen(language);
      std::string source = gen.GenApiFileDenseDecls(3000, params);

      if (!expected_bytes) {
        expected_bytes = source.size();
        expected_lines = CountLines(source);
        first_source = source;
        if (language == SourceGen::Language::Carbon) {
          EXPECT_TRUE(TestCompile(source));
        }
        continue;
      }
      EXPECT_THAT(source.size(), Eq(*expected_bytes))
          << "Byte count varied across seeds for language="
          << static_cast<int>(language);
      EXPECT_THAT(CountLines(source), Eq(*expected_lines))
          << "Line count varied across seeds for language="
          << static_cast<int>(language);
      if (source != *first_source) {
        any_different = true;
      }
    }
    EXPECT_TRUE(any_different);
  }
}

// Generated call graphs: bodies call the file's free function declarations,
// consuming each result into the accumulator. The line and byte counts must
// stay seed-independent, the result must compile cleanly, and the calls (and
// their `acc64` companion accumulator, whose presence is guaranteed by the
// deterministic count multisets) must actually appear.
TEST(SourceGenTest, GenApiFileDenseDeclsCallGraphs) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // The dense pattern with inline bodies making calls.
  param_set.push_back({.class_params = {.inline_function_defs = 3,
                                        .max_body_locals = 3,
                                        .max_body_calls = 3},
                       .free_function_decls_per_class = 2});
  // The split pattern: every body (inline and out-of-line) makes calls.
  param_set.push_back(
      {.class_params = {.inline_function_defs = 2, .max_body_calls = 2},
       .define_decls_out_of_line = true,
       .free_function_decls_per_class = 2});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      std::optional<size_t> expected_bytes;
      std::optional<ssize_t> expected_lines;
      std::optional<std::string> first_source;
      bool any_different = false;
      constexpr int NumSeeds = 16;
      for (int _ : llvm::seq(NumSeeds)) {
        SourceGen gen(language);
        std::string source = gen.GenApiFileDenseDecls(3000, params);
        if (!expected_bytes) {
          expected_bytes = source.size();
          expected_lines = CountLines(source);
          first_source = source;
          EXPECT_THAT(source, HasSubstr("acc64"));
          if (language == SourceGen::Language::Carbon) {
            EXPECT_TRUE(TestCompile(source));
          }
          continue;
        }
        EXPECT_THAT(source.size(), Eq(*expected_bytes))
            << "Byte count varied across seeds for language="
            << static_cast<int>(language);
        EXPECT_THAT(CountLines(source), Eq(*expected_lines))
            << "Line count varied across seeds for language="
            << static_cast<int>(language);
        if (source != *first_source) {
          any_different = true;
        }
      }
      EXPECT_TRUE(any_different);
    }
  }
}

// The line estimates must track the actual emission closely in every
// generation mode, or files drift away from their target size. Body-generating
// classes are large (hundreds of lines in the defined-decls pattern), so use a
// large target where the whole-class quantization of the file is small
// relative to the tolerance. Carbon is modeled tightly; C++ gets extra slack
// for its unmodeled access-section lines.
TEST(SourceGenTest, GenApiFileDenseDeclsLineTargetAccuracy) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // The benchmarked dense-declaration shape, with a couple of inline bodies
  // making calls to free functions.
  param_set.push_back(
      {.class_params = {.inline_function_defs = 2, .max_body_locals = 3},
       .free_function_decls_per_class = 2});
  // The benchmarked defined-decls shape.
  param_set.push_back(
      {.class_params = {.inline_function_defs = 2, .max_body_locals = 3},
       .define_decls_out_of_line = true,
       .free_function_decls_per_class = 2});

  constexpr int TargetLines = 20000;
  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      SourceGen gen(language);
      std::string source = gen.GenApiFileDenseDecls(TargetLines, params);
      ssize_t lines = CountLines(source);
      if (language == SourceGen::Language::Carbon) {
        // Within 2% of the requested line count.
        EXPECT_THAT(lines, AllOf(Ge(19600), Le(20400)))
            << "define_decls_out_of_line=" << params.define_decls_out_of_line;
      } else {
        // Within 10% of the requested line count.
        EXPECT_THAT(lines, AllOf(Ge(18000), Le(22000)))
            << "define_decls_out_of_line=" << params.define_decls_out_of_line;
      }
    }
  }
}

// Out-of-line definitions must also stay robust and deterministic for extreme,
// field-heavy class shapes, where most declared functions return non-copyable
// class types and there are few non-field slots.
TEST(SourceGenTest, GenApiFileDenseDeclsOutOfLineDefsRobust) {
  llvm::SmallVector<SourceGen::DenseDeclParams, 0> param_set;
  // Field-heavy classes with no inline definitions: every (few) declared
  // function/method is defined out-of-line, and the field type pool is built
  // entirely from fixed types.
  param_set.push_back({.class_params = {.public_function_decls = 2,
                                        .public_method_decls = 2,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 24,
                                        .inline_function_defs = 0},
                       .define_decls_out_of_line = true});
  // A mix of inline and out-of-line definitions with larger bodies.
  param_set.push_back({.class_params = {.public_function_decls = 4,
                                        .public_method_decls = 6,
                                        .private_function_decls = 2,
                                        .private_method_decls = 4,
                                        .private_field_decls = 6,
                                        .inline_function_defs = 4,
                                        .max_body_locals = 8},
                       .define_decls_out_of_line = true});

  for (const SourceGen::DenseDeclParams& params : param_set) {
    for (SourceGen::Language language :
         {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
      std::optional<size_t> expected_bytes;
      std::optional<ssize_t> expected_lines;
      constexpr int NumSeeds = 24;
      for (int _ : llvm::seq(NumSeeds)) {
        SourceGen gen(language);
        std::string source = gen.GenApiFileDenseDecls(3000, params);
        if (!expected_bytes) {
          expected_bytes = source.size();
          expected_lines = CountLines(source);
          if (language == SourceGen::Language::Carbon) {
            EXPECT_TRUE(TestCompile(source));
          }
          continue;
        }
        EXPECT_THAT(source.size(), Eq(*expected_bytes));
        EXPECT_THAT(CountLines(source), Eq(*expected_lines));
      }
    }
  }
}

}  // namespace
}  // namespace Carbon::Testing
