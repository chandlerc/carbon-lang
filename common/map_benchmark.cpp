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

template <typename MapT>
struct MapWrapper {
  using KeyT = typename MapT::key_type;
  using ValueT = typename MapT::mapped_type;

  MapT M;

  void CreateView() {}

  CARBON_ARM64_NOINLINE
  auto BenchContains(KeyT k) -> bool { return M.find(k) != M.end(); }

  CARBON_ARM64_NOINLINE
  auto BenchLookup(KeyT k) -> ValueT* {
    auto it = M.find(k);
    if (it == M.end()) {
      return nullptr;
    }
    ValueT* v = &it->second;
    return v;
  }

  CARBON_ARM64_NOINLINE
  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    return result.second;
  }

  CARBON_ARM64_NOINLINE
  auto BenchUpdate(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    result.first->second = v;
    return result.second;
  }

  CARBON_ARM64_NOINLINE
  auto BenchErase(KeyT k) -> bool { return M.erase(k) != 0; }
};

template <typename KT, typename VT, int MinSmallSize>
struct MapWrapper<Map<KT, VT, MinSmallSize>> {
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
};

// Helper to synthesize some value of one of the three types we use as value types.
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

static void OneOpSizeArgs(benchmark::internal::Benchmark* b) {
  b->DenseRange(1, 16, 4);
  b->DenseRange(24, 64, 8);
  b->Range(1 << 7, 1 << 20);
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

}  // namespace
}  // namespace Carbon
