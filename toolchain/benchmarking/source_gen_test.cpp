// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "toolchain/benchmarking/source_gen.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <optional>
#include <string>

#include "common/set.h"
#include "llvm/ADT/STLExtras.h"
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

// Check that the source code doesn't have compiler errors.
auto TestCompile(llvm::StringRef source) -> bool {
  llvm::IntrusiveRefCntPtr<llvm::vfs::InMemoryFileSystem> fs =
      new llvm::vfs::InMemoryFileSystem;
  InstallPaths installation(
      InstallPaths::MakeForBazelRunfiles(Testing::GetExePath()));
  Driver driver(fs, &installation, /*input_stream=*/nullptr, &llvm::outs(),
                &llvm::errs());

  AddPreludeFilesToVfs(installation, fs);

  fs->addFile("test.carbon", /*ModificationTime=*/0,
              llvm::MemoryBuffer::getMemBuffer(source));
  return driver
      .RunCommand({"compile", "--phase=check", "--no-include-carbon-core",
                   "test.carbon"})
      .success;
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
// actual content (identifiers, ordering) differs from run to run. This is what
// makes the generated code usable for stable, comparable benchmarking.
TEST(SourceGenTest, GenApiFileDenseDeclsStableSizeAcrossSeeds) {
  for (SourceGen::Language language :
       {SourceGen::Language::Carbon, SourceGen::Language::Cpp}) {
    // A spread of line targets large enough to fit at least one class up
    // through reasonably large files.
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
  // Many small functions with no parameters and no fields.
  param_set.push_back({.class_params = {.public_function_decls = 20,
                                        .public_method_decls = 0,
                                        .private_function_decls = 0,
                                        .private_method_decls = 0,
                                        .private_field_decls = 0}});
  // Methods with large parameter counts to exercise line-wrapping heuristics.
  param_set.push_back(
      {.class_params = {.public_function_decls = 2,
                        .public_function_decl_params = {.max_params = 16},
                        .public_method_decls = 4,
                        .public_method_decl_params = {.max_params = 16},
                        .private_function_decls = 0,
                        .private_method_decls = 0,
                        .private_field_decls = 0}});
  // The default shape, scaled up, keeping the default proportion of fields.
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
// class), so when most type uses are fields an unlucky shuffle could once leave
// the final class with no valid field type and crash `GetValidTypeName`. The
// generator caps how often each class is referenced to make this robust for any
// seed; this test guards that behavior across many seeds and extreme shapes,
// while also confirming the byte and line counts stay stable and the result
// still compiles.
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
  // as a type and every type use must fall back to a fixed type.
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
      // The historical failure was shuffle-dependent, so use many seeds.
      constexpr int NumSeeds = 32;
      for (int _ : llvm::seq(NumSeeds)) {
        SourceGen gen(language);
        std::string source = gen.GenApiFileDenseDecls(3000, params);
        if (!expected_bytes) {
          expected_bytes = source.size();
          expected_lines = CountLines(source);
          // The generated Carbon must remain valid: every emitted field type
          // must be a fixed type or an already-declared class.
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
