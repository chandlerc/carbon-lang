// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <memory>
#include <random>

#include "absl/container/flat_hash_set.h"
#include "common/set.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon {
namespace {

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
struct SetWrapper;

template <typename KT, int MinSmallSize>
struct SetWrapper<Set<KT, MinSmallSize>> {
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
};

template <typename KT, typename HasherT>
struct SetWrapper<absl::flat_hash_set<KT, HasherT>> {
  using SetT = absl::flat_hash_set<KT, HasherT>;
  using KeyT = KT;

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

template <typename KT, typename HasherT>
struct SetWrapper<llvm::DenseSet<KT, HasherT>> {
  using SetT = llvm::DenseSet<KT, HasherT>;
  using KeyT = KT;

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

template <typename KT, unsigned SmallSize, typename HasherT>
struct SetWrapper<llvm::SmallDenseSet<KT, SmallSize, HasherT>> {
  using SetT = llvm::SmallDenseSet<KT, SmallSize, HasherT>;
  using KeyT = KT;

  SetT M;

  void CreateView() {}

  auto BenchContains(KeyT k) -> bool {
    return benchmark::DoNotOptimize(M.find(k) != M.end());
  }

  auto BenchInsert(KeyT k) -> bool {
    auto result = M.insert(k);
    benchmark::DoNotOptimize(result.second);
    return result.second;
  }

  auto BenchErase(KeyT k) -> bool { return M.erase(k) != 0; }
};

struct CarbonHashingDenseInfo {
  // The following should hold, but it would require int to be complete:
  // static_assert(alignof(int) <= (1 << Log2MaxAlign),
  //               "DenseSet does not support pointer keys requiring more than "
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
    return static_cast<uint64_t>(HashValue(ptr_val));
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
  b->DenseRange(12, 16, 4);
  b->DenseRange(24, 64, 8);
  b->Range(1 << 7, 1 << 20);
}

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_ONE_OP(NAME)                             \
  BENCHMARK(NAME<Set<int*>>)->Apply(OneOpSizeArgs);                 \
  BENCHMARK(NAME<absl::flat_hash_set<int*>>)->Apply(OneOpSizeArgs); \
  BENCHMARK(NAME<llvm::DenseSet<int*, CarbonHashingDenseInfo>>)    \
      ->Apply(OneOpSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

template <typename SetT>
static void BM_SetContainsHitPtr(benchmark::State& s) {
  using SetWrapperT = SetWrapper<SetT>;
  SetWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key); });
  llvm::SmallVector<int*, 32> shuffled_keys = BuildShuffledKeys(keys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    bool result = m.BenchContains(shuffled_keys[i]);
    assert(result && "Should hit!");
    benchmark::DoNotOptimize(result);
    // We use `result` to step through the keys to establish a dependence
    // between each iteration of the loop and allow the benchmark to measure
    // *latency* rather than *throughput* -- the common case is to *do*
    // something with the result of a lookup.
    i = (i + static_cast<ssize_t>(result)) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetContainsHitPtr);

template <typename SetT>
static void BM_SetContainsMissPtr(benchmark::State& s) {
  using SetWrapperT = SetWrapper<SetT>;
  SetWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key); });
  constexpr ssize_t NumOtherKeys = 1024LL * 64;
  KeyVectorT other_keys = BuildKeys(NumOtherKeys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    bool result = m.BenchContains(other_keys[i].get());
    assert(!result && "Should miss!");
    benchmark::DoNotOptimize(result);
    // We use `result` to step through the keys to establish a dependence
    // between each iteration of the loop and allow the benchmark to measure
    // *latency* rather than *throughput* -- the common case is to *do*
    // something with the result of a lookup.
    i = (i + static_cast<ssize_t>(!result)) & (NumOtherKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetContainsMissPtr);

template <typename SetT>
static void BM_SetEraseInsertHitPtr(benchmark::State& s) {
  using SetWrapperT = SetWrapper<SetT>;
  SetWrapperT m;
  KeyVectorT keys =
      BuildKeys(s.range(0), [&m](int* key) { m.BenchInsert(key); });
  llvm::SmallVector<int*, 32> shuffled_keys = BuildShuffledKeys(keys);

  m.CreateView();

  ssize_t i = 0;
  for (auto _ : s) {
    m.BenchErase(shuffled_keys[i]);
    benchmark::ClobberMemory();
    bool inserted = m.BenchInsert(shuffled_keys[i]);
    assert(inserted && "Should insert!");
    benchmark::DoNotOptimize(inserted);
    // We use `inserted` to step through the keys to establish a dependence
    // between each iteration of the loop and allow the benchmark to measure
    // *latency* rather than *throughput* -- the common case is to *do*
    // something with the result of a lookup.
    i = (i + static_cast<ssize_t>(inserted)) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_SetEraseInsertHitPtr);

static void OpSeqSizeArgs(benchmark::internal::Benchmark* b) {
  b->DenseRange(1, 13, 1);
  b->DenseRange(15, 17, 1);
  b->DenseRange(23, 25, 1);
  b->DenseRange(31, 33, 1);
  b->Range(1 << 6, 1 << 15);
}

// NOLINTBEGIN(bugprone-macro-parentheses): Parentheses are incorrect here.
#define MAP_BENCHMARK_OP_SEQ(NAME)                             \
  BENCHMARK(NAME<Set<int*>>)->Apply(OpSeqSizeArgs);                 \
  BENCHMARK(NAME<absl::flat_hash_set<int*>>)->Apply(OpSeqSizeArgs); \
  BENCHMARK(NAME<llvm::DenseSet<int*, CarbonHashingDenseInfo>>)    \
      ->Apply(OpSeqSizeArgs)
// NOLINTEND(bugprone-macro-parentheses)

template <typename SetT>
static void BM_SetInsertPtrSeq(benchmark::State& s) {
  using SetWrapperT = SetWrapper<SetT>;
  KeyVectorT keys = BuildKeys(s.range(0));
  llvm::SmallVector<int*, 0> shuffled_keys = BuildShuffledKeys(keys);

  ssize_t i = 0;
  for (auto _ : s) {
    // First insert all the keys.
    SetWrapperT m;
    for (const auto& k : keys) {
      bool inserted = m.BenchInsert(k.get());
      assert(inserted && "Must be a successful insert!");
      (void)inserted;
    }

    // Now insert a final random repeated key.
    bool inserted = m.BenchInsert(shuffled_keys[i]);
    assert(!inserted && "Must already be in the map!");
    (void)inserted;

    // Rotate through the shuffled keys.
    i = (i + 1) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_OP_SEQ(BM_SetInsertPtrSeq);

}  // namespace
}  // namespace Carbon
