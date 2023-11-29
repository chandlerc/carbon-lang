// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include "absl/container/flat_hash_set.h"
#include "common/raw_hashtable_benchmark_helpers.h"
#include "common/set.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon {
namespace {

using RawHashtable::BuildKeys;
using RawHashtable::BuildShuffledKeys;
using RawHashtable::CarbonHashingDenseInfo;
using RawHashtable::NumOtherKeys;
using RawHashtable::NumShuffledKeys;
using RawHashtable::OneOpSizeArgs;
using RawHashtable::OpSeqSizeArgs;

// This map has an intentional inlining blocker to avoid code growth. However,
// both Abseil and LLVM's maps don't have this and at least on AArch64 both
// inline heavily and show performance differences that seem entirely to stem
// from that. We use this to block inlining on the wrapper as a way to get
// slightly more comparable benchmark results. It's not perfect, but seems more
// useful than the alternatives.
#ifdef __aarch64__
#define CARBON_ARM64_NOINLINE [[gnu::noinline]]
#else
#define CARBON_ARM64_NOINLINE
#endif

template <typename SetT>
struct SetWrapper {
  static constexpr bool IsCarbonSet = false;
  using KeyT = typename SetT::key_type;

  SetT M;

  void CreateView() {}

  CARBON_ARM64_NOINLINE
  auto BenchContains(KeyT k) -> bool { return M.find(k) != M.end(); }

  CARBON_ARM64_NOINLINE
  auto BenchInsert(KeyT k) -> bool {
    auto result = M.insert(k);
    benchmark::DoNotOptimize(result.second);
    return result.second;
  }

  CARBON_ARM64_NOINLINE
  auto BenchErase(KeyT k) -> bool { return M.erase(k) != 0; }
};

template <typename KT, int MinSmallSize>
struct SetWrapper<Set<KT, MinSmallSize>> {
  static constexpr bool IsCarbonSet = true;
  using SetT = Set<KT, MinSmallSize>;
  using KeyT = KT;

  SetT M;

  SetView<KT> MV = M;

  void CreateView() { MV = M; }

  auto BenchContains(KeyT k) -> bool { return MV.Contains(k); }

  auto BenchInsert(KeyT k) -> bool {
    auto result = M.Insert(k);
    benchmark::DoNotOptimize(result.is_inserted());
    return result.is_inserted();
  }

  auto BenchErase(KeyT k) -> bool { return M.Erase(k); }

  auto CountProbedKeys() const -> ssize_t { return M.CountProbedKeys(); }
};

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_ONE_OP_SIZE(NAME, KT)                       \
  BENCHMARK(NAME<Set<KT>>)->Apply(OneOpSizeArgs);                 \
  BENCHMARK(NAME<absl::flat_hash_set<KT>>)->Apply(OneOpSizeArgs); \
  BENCHMARK(NAME<llvm::DenseSet<KT, CarbonHashingDenseInfo<KT>>>) \
      ->Apply(OneOpSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_ONE_OP(NAME) MAP_BENCHMARK_ONE_OP_SIZE(NAME, int*)

template <typename SetT>
static void BM_SetContainsHitPtr(benchmark::State& s) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  SetWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0));
  for (auto k : keys) {
    m.BenchInsert(k);
  }
  llvm::SmallVector<KT> shuffled_keys = BuildShuffledKeys(keys);
  ssize_t i = 0;

  m.CreateView();
  for (auto _ : s) {
    bool result = m.BenchContains(shuffled_keys[i]);
    CARBON_DCHECK(result);
    // We use the lookup success to step through keys, establishing a dependency
    // between each lookup and allowing us to measure latency rather than
    // throughput.
    i = (i + static_cast<ssize_t>(result)) & (NumShuffledKeys - 1);
    // We block optimizing `i` as that has proven both more effective at
    // blocking the loop from being optimized away and avoiding disruption of
    // the generated code that we're benchmarking.
    benchmark::DoNotOptimize(i);
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetContainsHitPtr);

template <typename SetT>
static void BM_SetContainsMissPtr(benchmark::State& s) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  SetWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0) + NumOtherKeys);
  for (auto k : keys.slice(0, s.range(0))) {
    m.BenchInsert(k);
  }
  llvm::SmallVector<KT> shuffled_keys =
      BuildShuffledKeys(keys.slice(s.range(0)));
  ssize_t i = 0;

  m.CreateView();
  for (auto _ : s) {
    bool result = m.BenchContains(shuffled_keys[i]);
    CARBON_DCHECK(!result);
    i = (i + static_cast<ssize_t>(!result)) & (NumShuffledKeys - 1);
    benchmark::DoNotOptimize(i);
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetContainsMissPtr);

template <typename SetT>
static void BM_SetEraseInsertHitPtr(benchmark::State& s) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  SetWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0));
  for (auto k : keys) {
    m.BenchInsert(k);
  }
  llvm::SmallVector<KT> shuffled_keys = BuildShuffledKeys(keys);
  ssize_t i = 0;

  m.CreateView();
  for (auto _ : s) {
    m.BenchErase(shuffled_keys[i]);
    benchmark::ClobberMemory();
    bool inserted = m.BenchInsert(shuffled_keys[i]);
    CARBON_DCHECK(inserted);
    i = (i + static_cast<ssize_t>(inserted)) & (NumShuffledKeys - 1);
    benchmark::DoNotOptimize(i);
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetEraseInsertHitPtr);

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_OP_SEQ_SIZE(NAME, KT)                       \
  BENCHMARK(NAME<Set<KT>>)->Apply(OpSeqSizeArgs);                 \
  BENCHMARK(NAME<absl::flat_hash_set<KT>>)->Apply(OpSeqSizeArgs); \
  BENCHMARK(NAME<llvm::DenseSet<KT, CarbonHashingDenseInfo<KT>>>) \
      ->Apply(OpSeqSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_OP_SEQ(NAME) MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int*)

template <typename SetT>
static void BM_SetInsertPtrSeq(benchmark::State& s) {
  using SetWrapperT = SetWrapper<SetT>;
  using KT = typename SetWrapperT::KeyT;
  llvm::ArrayRef<KT> raw_keys = BuildKeys<KT>(s.range(0));
  // We want to permute the keys so they're not in any particular order, but not
  // generate a full shuffled set. Note that the branch predictor is likely to
  // still be able to learn the complete pattern of branches inserted for very
  // small key ranges.
  llvm::SmallVector<KT> keys(raw_keys.begin(), raw_keys.end());
  std::shuffle(keys.begin(), keys.end(), absl::BitGen());

  // Now build a large shuffled set of keys (with duplicates) we'll use at the
  // end.
  llvm::SmallVector<KT> shuffled_keys = BuildShuffledKeys(raw_keys);
  ssize_t i = 0;
  for (auto _ : s) {
    SetWrapperT m;
    for (auto k : keys) {
      bool inserted = m.BenchInsert(k);
      CARBON_DCHECK(inserted) << "Must be a successful insert!";
    }

    // Now insert a final random repeated key.
    bool inserted = m.BenchInsert(shuffled_keys[i]);
    CARBON_DCHECK(!inserted) << "Must already be in the map!";

    // Rotate through the shuffled keys.
    i = (i + static_cast<ssize_t>(inserted)) & (NumShuffledKeys - 1);
    benchmark::DoNotOptimize(i);
  }

  // It can be easier in some cases to think of this as a key-throughput rate of
  // insertion rather than the latency of inserting N keys, so construct the
  // rate counter as well.
  s.counters["KeyRate"] = benchmark::Counter(
      keys.size(), benchmark::Counter::kIsIterationInvariantRate);

  // Report some extra statistics about the Carbon type.
  if constexpr (SetWrapperT::IsCarbonSet) {
    // Re-build a set outside of the timing loop to look at the statistics
    // rather than the timing.
    SetWrapperT set;
    for (auto k : keys) {
      bool inserted = set.BenchInsert(k);
      CARBON_DCHECK(inserted) << "Must be a successful insert!";
    }

    // While this count is "iteration invariant" (it should be exactly the same
    // for every iteration as the set of keys is the same), we don't use that
    // because it will scale this by the number of iterations. We want to
    // display the probe count of this benchmark *parameter*, not the probe
    // count that resulted from the number of iterations. That means we use the
    // normal counter API without flags.
    s.counters["NumProbed"] = benchmark::Counter(set.CountProbedKeys());

    // Uncomment this call to print out statistics about the index-collisions
    // among these keys for debugging:
    //
    // RawHashtable::DumpHashStatistics(raw_keys);
  }
}
MAP_BENCHMARK_OP_SEQ(BM_SetInsertPtrSeq);

}  // namespace
}  // namespace Carbon
