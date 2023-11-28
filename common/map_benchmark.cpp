// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "common/map.h"
#include "common/raw_hashtable_benchmark_helpers.h"
#include "llvm/ADT/DenseMap.h"
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

template <typename MapT>
struct MapWrapper {
  static constexpr bool IsCarbonMap = false;
  using KeyT = typename MapT::key_type;
  using ValueT = typename MapT::mapped_type;

  MapT M;

  void CreateView() {}

  auto BenchContains(KeyT k) -> bool { return M.find(k) != M.end(); }

  auto BenchLookup(KeyT k) -> ValueT* {
    auto it = M.find(k);
    if (it == M.end()) {
      return nullptr;
    }
    ValueT* v = &it->second;
    return v;
  }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    return result.second;
  }

  auto BenchUpdate(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    result.first->second = v;
    return result.second;
  }

  auto BenchErase(KeyT k) -> bool { return M.erase(k) != 0; }
};

template <typename KT, typename VT, int MinSmallSize>
struct MapWrapper<Map<KT, VT, MinSmallSize>> {
  static constexpr bool IsCarbonMap = true;
  using MapT = Map<KT, VT, MinSmallSize>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  MapView<KT, VT> MV = M;

  void CreateView() { MV = M; }

  auto BenchLookup(KeyT k) -> ValueT* {
    ValueT* v = MV[k];
    return v;
  }

  auto BenchContains(KeyT k) -> bool { return MV.Contains(k); }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.Insert(k, v);
    return result.is_inserted();
  }

  auto BenchUpdate(KeyT k, ValueT v) -> bool {
    auto result = M.Update(k, v);
    return result.is_inserted();
  }

  auto BenchErase(KeyT k) -> bool { return M.Erase(k); }

  auto CountProbedKeys() const -> ssize_t { return M.CountProbedKeys(); }
};

// Helper to synthesize some value of one of the three types we use as value
// types.
template <typename T>
auto MakeValue2() -> T {
  if constexpr (std::is_same_v<T, llvm::StringRef>) {
    return "abc";
  } else if constexpr (std::is_pointer_v<T>) {
    static std::remove_pointer_t<T> x;
    return &x;
  } else {
    return 42;
  }
}

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_ONE_OP_SIZE(NAME, KT, VT)                       \
  BENCHMARK(NAME<Map<KT, VT>>)->Apply(OneOpSizeArgs);                 \
  BENCHMARK(NAME<absl::flat_hash_map<KT, VT>>)->Apply(OneOpSizeArgs); \
  BENCHMARK(NAME<llvm::DenseMap<KT, VT, CarbonHashingDenseInfo<KT>>>) \
      ->Apply(OneOpSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_ONE_OP(NAME)                       \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, int, int);             \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, int*, int*);           \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, int, llvm::StringRef); \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, llvm::StringRef, int);

template <typename MapT>
static void BM_MapContainsHit(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0));
  for (auto k : keys) {
    m.BenchInsert(k, VT());
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
MAP_BENCHMARK_ONE_OP(BM_MapContainsHit);

template <typename MapT>
static void BM_MapContainsMiss(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0) + NumOtherKeys);
  for (auto k : keys.slice(0, s.range(0))) {
    m.BenchInsert(k, VT());
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
MAP_BENCHMARK_ONE_OP(BM_MapContainsMiss);

template <typename MapT>
static void BM_MapLookupHit(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0));
  for (auto k : keys) {
    m.BenchInsert(k, VT());
  }
  llvm::SmallVector<KT> shuffled_keys = BuildShuffledKeys(keys);
  ssize_t i = 0;

  m.CreateView();
  for (auto _ : s) {
    VT* value = m.BenchLookup(shuffled_keys[i]);
    CARBON_DCHECK(value != nullptr);
    i = (i + static_cast<ssize_t>(value != nullptr)) & (NumShuffledKeys - 1);
    benchmark::DoNotOptimize(i);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapLookupHit);

template <typename MapT>
static void BM_MapLookupMiss(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0) + NumOtherKeys);
  for (auto k : keys.slice(0, s.range(0))) {
    m.BenchInsert(k, VT());
  }
  llvm::SmallVector<KT> shuffled_keys =
      BuildShuffledKeys(keys.slice(s.range(0)));
  ssize_t i = 0;

  m.CreateView();
  for (auto _ : s) {
    VT* value = m.BenchLookup(shuffled_keys[i]);
    CARBON_DCHECK(value == nullptr);
    i = (i + static_cast<ssize_t>(value == nullptr)) & (NumOtherKeys - 1);
    benchmark::DoNotOptimize(i);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapLookupMiss);

template <typename MapT>
static void BM_MapUpdateHit(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0));
  for (auto k : keys) {
    m.BenchInsert(k, VT());
  }
  llvm::SmallVector<KT> shuffled_keys = BuildShuffledKeys(keys);
  ssize_t i = 0;

  m.CreateView();
  for (auto _ : s) {
    benchmark::ClobberMemory();
    bool inserted = m.BenchUpdate(shuffled_keys[i], MakeValue2<VT>());
    CARBON_DCHECK(!inserted);
    benchmark::DoNotOptimize(inserted);
    i = (i + static_cast<ssize_t>(!inserted)) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapUpdateHit);

template <typename MapT>
static void BM_MapEraseUpdateHit(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
  llvm::ArrayRef<KT> keys = BuildKeys<KT>(s.range(0));
  for (auto k : keys) {
    m.BenchInsert(k, VT());
  }
  llvm::SmallVector<KT> shuffled_keys = BuildShuffledKeys(keys);
  ssize_t i = 0;

  m.CreateView();
  for (auto _ : s) {
    m.BenchErase(shuffled_keys[i]);
    benchmark::ClobberMemory();
    bool inserted = m.BenchUpdate(shuffled_keys[i], MakeValue2<VT>());
    CARBON_DCHECK(inserted);
    i = (i + static_cast<ssize_t>(inserted)) & (NumShuffledKeys - 1);
    benchmark::DoNotOptimize(i);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapEraseUpdateHit);

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_OP_SEQ_SIZE(NAME, KT, VT)                       \
  BENCHMARK(NAME<Map<KT, VT>>)->Apply(OpSeqSizeArgs);                 \
  BENCHMARK(NAME<absl::flat_hash_map<KT, VT>>)->Apply(OpSeqSizeArgs); \
  BENCHMARK(NAME<llvm::DenseMap<KT, VT, CarbonHashingDenseInfo<KT>>>) \
      ->Apply(OpSeqSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_OP_SEQ(NAME)                       \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int, int);             \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int*, int*);           \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, int, llvm::StringRef); \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, llvm::StringRef, int)

template <typename MapT>
static void BM_MapInsertSeq(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using KT = typename MapWrapperT::KeyT;
  using VT = typename MapWrapperT::ValueT;
  MapWrapperT m;
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
    MapWrapperT m;
    for (auto k : keys) {
      bool inserted = m.BenchInsert(k, MakeValue2<VT>());
      CARBON_DCHECK(inserted) << "Must be a successful insert!";
    }

    // Now insert a final random repeated key.
    bool inserted = m.BenchInsert(shuffled_keys[i], MakeValue2<VT>());
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
  if constexpr (MapWrapperT::IsCarbonMap) {
    // Re-build a map outside of the timing loop to look at the statistics
    // rather than the timing.
    MapWrapperT map;
    for (auto k : keys) {
      bool inserted = map.BenchInsert(k, MakeValue2<VT>());
      CARBON_DCHECK(inserted) << "Must be a successful insert!";
    }

    // While this count is "iteration invariant" (it should be exactly the same
    // for every iteration as the set of keys is the same), we don't use that
    // because it will scale this by the number of iterations. We want to
    // display the probe count of this benchmark *parameter*, not the probe
    // count that resulted from the number of iterations. That means we use the
    // normal counter API without flags.
    s.counters["NumProbed"] = benchmark::Counter(map.CountProbedKeys());
  }
}
MAP_BENCHMARK_OP_SEQ(BM_MapInsertSeq);

}  // namespace
}  // namespace Carbon
