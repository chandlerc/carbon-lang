// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <memory>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "common/map.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {
namespace {

// This map has an intentional inlining blocker to avoid code growth. However,
// both Abseil and LLVM's maps don't have this and at least on AArch64 both
// inline heavily and show performance differences that seem entirely to stem
// from that. We use this to block inlining on the wrapper as a way to get
// slightly more comparable benchmark results. It's not perfect, but seems more
// useful than the alternatives.
#ifdef __aarch64__
#define CARBON_ARM64_NOINLINE [[clang::noinline]]
#else
#define CARBON_ARM64_NOINLINE
#endif

template <typename MapT>
struct MapWrapper;

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
    benchmark::DoNotOptimize(v);
    return v;
  }

  auto BenchContains(KeyT k) -> bool { return MV.Contains(k); }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.Insert(k, v);
    benchmark::DoNotOptimize(result.is_inserted());
    return result.is_inserted();
  }

  auto BenchUpdate(KeyT k, ValueT v) -> void { M.Update(k, v); }

  auto BenchErase(KeyT k) -> bool { return M.Erase(k); }
};

template <typename KT, typename VT, typename HasherT>
struct MapWrapper<absl::flat_hash_map<KT, VT, HasherT>> {
  using MapT = absl::flat_hash_map<KT, VT, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  void CreateView() {}

  CARBON_ARM64_NOINLINE
  auto BenchContains(KeyT k) -> bool { return M.find(k) != M.end(); }

  CARBON_ARM64_NOINLINE
  auto BenchLookup(KeyT k) -> ValueT* {
    auto it = M.find(k);
    ValueT* v = &it->second;
    benchmark::DoNotOptimize(v);
    return v;
  }

  CARBON_ARM64_NOINLINE
  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    benchmark::DoNotOptimize(result.second);
    return result.second;
  }

  CARBON_ARM64_NOINLINE
  auto BenchUpdate(KeyT k, ValueT v) -> void { M[k] = v; }

  CARBON_ARM64_NOINLINE
  auto BenchErase(KeyT k) -> bool { return M.erase(k) != 0; }
};

template <typename KT, typename VT, typename HasherT>
struct MapWrapper<llvm::DenseMap<KT, VT, HasherT>> {
  using MapT = llvm::DenseMap<KT, VT, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  void CreateView() {}

  CARBON_ARM64_NOINLINE
  auto BenchContains(KeyT k) -> bool { return M.find(k) != M.end(); }

  CARBON_ARM64_NOINLINE
  auto BenchLookup(KeyT k) -> ValueT* {
    auto it = M.find(k);
    ValueT* v = &it->second;
    benchmark::DoNotOptimize(v);
    return v;
  }

  CARBON_ARM64_NOINLINE
  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    benchmark::DoNotOptimize(result.second);
    return result.second;
  }

  CARBON_ARM64_NOINLINE
  auto BenchUpdate(KeyT k, ValueT v) -> void { M[k] = v; }

  CARBON_ARM64_NOINLINE
  auto BenchErase(KeyT k) -> bool { return M.erase(k) != 0; }
};

template <typename KT, typename VT, unsigned SmallSize, typename HasherT>
struct MapWrapper<llvm::SmallDenseMap<KT, VT, SmallSize, HasherT>> {
  using MapT = llvm::SmallDenseMap<KT, VT, SmallSize, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  void CreateView() {}

  auto BenchContains(KeyT k) -> bool {
    return benchmark::DoNotOptimize(M.find(k) != M.end());
  }

  auto BenchLookup(KeyT k) -> bool {
    auto it = M.find(k);
    ValueT* v = &it->second;
    benchmark::DoNotOptimize(v);
    return v;
  }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    benchmark::DoNotOptimize(result.second);
    return result.second;
  }

  auto BenchUpdate(KeyT k, ValueT v) -> void { M[k] = v; }

  auto BenchErase(KeyT k) -> bool { return M.erase(k) != 0; }
};

struct LLVMHash {
  template <typename T>
  auto operator()(const T& arg) const -> size_t {
    using llvm::hash_value;
    return hash_value(arg);
  }
};

struct LLVMHashingDenseMapInfo {
  // The following should hold, but it would require int to be complete:
  // static_assert(alignof(int) <= (1 << Log2MaxAlign),
  //               "DenseMap does not support pointer keys requiring more than "
  //               "Log2MaxAlign bits of alignment");
  static constexpr uintptr_t Log2MaxAlign = 12;

  static inline auto getEmptyKey() -> int* {
    auto val = static_cast<uintptr_t>(-1);
    val <<= Log2MaxAlign;
    // NOLINTNEXTLINE(performance-no-int-to-ptr): This is required by the API.
    return reinterpret_cast<int*>(val);
  }

  static inline auto getTombstoneKey() -> int* {
    auto val = static_cast<uintptr_t>(-2);
    val <<= Log2MaxAlign;
    // NOLINTNEXTLINE(performance-no-int-to-ptr): This is required by the API.
    return reinterpret_cast<int*>(val);
  }

  static auto getHashValue(const int* ptr_val) -> unsigned {
    using llvm::hash_value;
    return hash_value(ptr_val);
  }

  static auto isEqual(const int* lhs, const int* rhs) -> bool {
    return lhs == rhs;
  }
};

using KeyVectorT = llvm::ArrayRef<std::unique_ptr<int>>;

[[clang::noinline]] auto BuildKeys(
    ssize_t size, llvm::function_ref<void(int*)> callback = [](int*) {})
    -> KeyVectorT {
  constexpr ssize_t MaxKeysSize = 1 << 20;
  static std::vector<std::unique_ptr<int>>& keys = *([] {
    auto* keys_ptr = new std::vector<std::unique_ptr<int>>();
    for (ssize_t i : llvm::seq<ssize_t>(0, MaxKeysSize)) {
      keys_ptr->emplace_back(new int(i));
    }
    return keys_ptr;
  }());

  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    callback(keys[i].get());
  }

  return llvm::ArrayRef(keys).take_front(size);
}

constexpr ssize_t NumShuffledKeys = 1024LL * 64;

[[clang::noinline]] auto BuildShuffledKeys(const KeyVectorT& keys)
    -> llvm::SmallVector<int*, 0> {
  std::random_device r_dev;
  std::seed_seq seed(
      {r_dev(), r_dev(), r_dev(), r_dev(), r_dev(), r_dev(), r_dev(), r_dev()});
  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> d(0, keys.size() - 1);

  llvm::SmallVector<int*, 32> shuffled_keys;
  for (ssize_t i : llvm::seq<ssize_t>(0, NumShuffledKeys)) {
    (void)i;
    ssize_t random_idx = d(rng);
    assert(random_idx < (ssize_t)keys.size() && "Too large value!");
    shuffled_keys.push_back(keys[random_idx].get());
  }

  return shuffled_keys;
}

static void OneOpSizeArgs(benchmark::internal::Benchmark* b) {
  b->DenseRange(1, 8, 1);
  b->DenseRange(10, 16, 2);
  b->DenseRange(24, 64, 8);
  b->Range(1 << 7, 1 << 20);
}

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_ONE_OP_SIZE(NAME, SIZE)                                 \
  BENCHMARK(NAME<Map<int*, std::array<int, SIZE>>>)->Apply(OneOpSizeArgs);    \
  BENCHMARK(NAME<absl::flat_hash_map<int*, std::array<int, SIZE>, LLVMHash>>) \
      ->Apply(OneOpSizeArgs);                                                 \
  BENCHMARK(NAME<llvm::DenseMap<int*, std::array<int, SIZE>,                  \
                                LLVMHashingDenseMapInfo>>)                    \
      ->Apply(OneOpSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_ONE_OP(NAME)    \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, 1); \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, 2); \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, 4); \
  MAP_BENCHMARK_ONE_OP_SIZE(NAME, 64)

template <typename MapT>
static void BM_MapLookupHitPtr(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  MapWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key, T()); });
  llvm::SmallVector<int*, 32> shuffled_keys = BuildShuffledKeys(keys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    T* value = m.BenchLookup(shuffled_keys[i]);
    assert(value && "Lookup must succeed!");
    (void)value;
    i = (i + 1) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapLookupHitPtr);

template <typename MapT>
static void BM_MapLookupMissPtr(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  MapWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key, T()); });
  constexpr ssize_t NumOtherKeys = 1024LL * 64;
  KeyVectorT other_keys = BuildKeys(NumOtherKeys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    T* value = m.BenchLookup(other_keys[i].get());
    assert(!value && "Lookup must fail!");
    (void)value;
    i = (i + 1) & (NumOtherKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapLookupMissPtr);

template <typename MapT>
static void BM_MapContainsHitPtr(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  MapWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key, T()); });
  llvm::SmallVector<int*, 32> shuffled_keys = BuildShuffledKeys(keys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    bool result = m.BenchContains(shuffled_keys[i]);
    assert(result && "Should hit!");
    benchmark::DoNotOptimize(result);
    i = (i + 1) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapContainsHitPtr);

template <typename MapT>
static void BM_MapContainsMissPtr(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  MapWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key, T()); });
  constexpr ssize_t NumOtherKeys = 1024LL * 64;
  KeyVectorT other_keys = BuildKeys(NumOtherKeys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    bool result = m.BenchContains(other_keys[i].get());
    assert(!result && "Should miss!");
    benchmark::DoNotOptimize(result);
    i = (i + 1) & (NumOtherKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapContainsMissPtr);

template <typename MapT>
static void BM_MapUpdateHitPtr(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  MapWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key, T()); });
  llvm::SmallVector<int*, 32> shuffled_keys = BuildShuffledKeys(keys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    benchmark::ClobberMemory();
    m.BenchUpdate(shuffled_keys[i], {});
    i = (i + 1) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapUpdateHitPtr);

template <typename MapT>
static void BM_MapEraseUpdateHitPtr(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  MapWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key, T()); });
  llvm::SmallVector<int*, 32> shuffled_keys = BuildShuffledKeys(keys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    m.BenchErase(shuffled_keys[i]);
    benchmark::ClobberMemory();
    m.BenchUpdate(shuffled_keys[i], {});
    i = (i + 1) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapEraseUpdateHitPtr);

static void OpSeqSizeArgs(benchmark::internal::Benchmark* b) {
  b->DenseRange(1, 13, 1);
  b->DenseRange(15, 17, 1);
  b->DenseRange(23, 25, 1);
  b->DenseRange(31, 33, 1);
  b->Range(1 << 6, 1 << 15);
}

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_OP_SEQ_SIZE(NAME, SIZE)                                 \
  BENCHMARK(NAME<Map<int*, std::array<int, SIZE>>>)->Apply(OpSeqSizeArgs);    \
  BENCHMARK(NAME<absl::flat_hash_map<int*, std::array<int, SIZE>, LLVMHash>>) \
      ->Apply(OpSeqSizeArgs);                                                 \
  BENCHMARK(NAME<llvm::DenseMap<int*, std::array<int, SIZE>,                  \
                                LLVMHashingDenseMapInfo>>)                    \
      ->Apply(OpSeqSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_OP_SEQ(NAME)    \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, 1); \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, 2); \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, 4); \
  MAP_BENCHMARK_OP_SEQ_SIZE(NAME, 64)

template <typename MapT>
static void BM_MapInsertPtrSeq(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  KeyVectorT keys = BuildKeys(s.range(0));
  llvm::SmallVector<int*, 0> shuffled_keys = BuildShuffledKeys(keys);

  ssize_t i = 0;
  for (auto _ : s) {
    // First insert all the keys.
    MapWrapperT m;
    for (const auto& k : keys) {
      bool inserted = m.BenchInsert(k.get(), T());
      assert(inserted && "Must be a successful insert!");
      (void)inserted;
    }

    // Now insert a final random repeated key.
    bool inserted = m.BenchInsert(shuffled_keys[i], T());
    assert(!inserted && "Must already be in the map!");
    (void)inserted;

    // Rotate through the shuffled keys.
    i = (i + 1) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_OP_SEQ(BM_MapInsertPtrSeq);

}  // namespace
}  // namespace Carbon::Testing
