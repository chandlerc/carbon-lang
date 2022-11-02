// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <array>
#include <cstddef>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "common/map.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {
namespace {

template <typename MapT>
struct MapWrapper;

template <typename KT, typename VT, int MinSmallSize>
struct MapWrapper<Map<KT, VT, MinSmallSize>> {
  using MapT = Map<KT, VT, MinSmallSize>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  auto BenchLookup(KeyT k) -> bool {
    auto* v = M[k];
    benchmark::DoNotOptimize(v);
    return v != nullptr;
  }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert(k, v);
    benchmark::DoNotOptimize(result.isInserted());
    return result.isInserted();
  }
};

template <typename KT, typename VT, typename HasherT>
struct MapWrapper<absl::flat_hash_map<KT, VT, HasherT>> {
  using MapT = absl::flat_hash_map<KT, VT, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  auto BenchLookup(KeyT k) -> bool {
    auto it = M.find(k);
    benchmark::DoNotOptimize(it);
    return it != M.end();
  }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    benchmark::DoNotOptimize(result.second);
    return result.second;
  }
};

template <typename KT, typename VT, typename HasherT>
struct MapWrapper<llvm::DenseMap<KT, VT, HasherT>> {
  using MapT = llvm::DenseMap<KT, VT, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  auto BenchLookup(KeyT k) -> bool {
    auto it = M.find(k);
    benchmark::DoNotOptimize(it);
    return it != M.end();
  }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    benchmark::DoNotOptimize(result.second);
    return result.second;
  }
};

template <typename KT, typename VT, unsigned SmallSize, typename HasherT>
struct MapWrapper<llvm::SmallDenseMap<KT, VT, SmallSize, HasherT>> {
  using MapT = llvm::SmallDenseMap<KT, VT, SmallSize, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  auto BenchLookup(KeyT k) -> bool {
    auto it = M.find(k);
    benchmark::DoNotOptimize(it);
    return it != M.end();
  }

  auto BenchInsert(KeyT k, ValueT v) -> bool {
    auto result = M.insert({k, v});
    benchmark::DoNotOptimize(result.second);
    return result.second;
  }
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

using KeyVectorT = llvm::SmallVector<std::unique_ptr<int>, 32>;

auto BuildKeys(
    ssize_t size, llvm::function_ref<void(int*)> callback = [](int*) {})
    -> KeyVectorT {
  KeyVectorT keys;
  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    keys.emplace_back(new int(i));
  }

  for (ssize_t i : llvm::seq<ssize_t>(0, size)) {
    callback(keys[i].get());
  }

  return keys;
}

constexpr ssize_t NumShuffledKeys = 1024LL * 64;

auto BuildShuffledKeys(const KeyVectorT& keys) -> llvm::SmallVector<int*, 32> {
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

static void OneOpSizeArgs(benchmark::internal::Benchmark *b) {
    b->Arg(1);
    b->Arg(2);
    b->Arg(4);
    b->Arg(8);
    b->Arg(16);
    b->Arg(32);
    b->Range(1 << 6, 1 << 20);
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

#define MAP_BENCHMARK_ONE_OP(NAME) \
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

  ssize_t i = 0;
  for (auto _ : s) {
    bool result = m.BenchLookup(shuffled_keys[i]);
    assert(result && "Lookup must succeed!");
    (void)result;
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

  ssize_t i = 0;
  for (auto _ : s) {
    bool result = m.BenchLookup(other_keys[i].get());
    assert(!result && "Lookup must fail!");
    (void)result;
    i = (i + 1) & (NumOtherKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapLookupMissPtr);

static void OpSeqSizeArgs(benchmark::internal::Benchmark *b) {
    b->Arg(1);
    b->Arg(2);
    b->Arg(3);
    b->Arg(4);
    b->Arg(5);
    b->Arg(8);
    b->Arg(9);
    b->Arg(16);
    b->Arg(17);
    b->Arg(32);
    b->Arg(33);
    b->Range(1 << 6, 1 << 15);
}

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_OP_SEQ_SIZE(NAME, SIZE)                                 \
  BENCHMARK(NAME<Map<int*, std::array<int, SIZE>>>)->Apply(OpSeqSizeArgs);    \
  BENCHMARK(NAME<absl::flat_hash_map<int*, std::array<int, SIZE>, LLVMHash>>) \
      ->Apply(OneOpSizeArgs);                                                 \
  BENCHMARK(NAME<llvm::DenseMap<int*, std::array<int, SIZE>,                  \
                                LLVMHashingDenseMapInfo>>)                    \
      ->Apply(OneOpSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

#define MAP_BENCHMARK_OP_SEQ(NAME) \
    MAP_BENCHMARK_OP_SEQ_SIZE(NAME, 1); \
    MAP_BENCHMARK_OP_SEQ_SIZE(NAME, 2); \
    MAP_BENCHMARK_OP_SEQ_SIZE(NAME, 4); \
    MAP_BENCHMARK_OP_SEQ_SIZE(NAME, 64)

template <typename MapT>
static void BM_MapInsertPtrSeq(benchmark::State& s) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  KeyVectorT keys = BuildKeys(s.range(0));
  llvm::SmallVector<int*, 32> shuffled_keys = BuildShuffledKeys(keys);

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
