// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <array>
#include <random>

#include "absl/container/flat_hash_map.h"
#include "common/map.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

namespace Carbon::Testing {
namespace {

template <typename MapT> struct MapWrapper;

template <typename KT, typename VT, int MinSmallSize>
struct MapWrapper<Map<KT, VT, MinSmallSize>> {
  using MapT = Map<KT, VT, MinSmallSize>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  bool BenchLookup(KeyT K) {
    auto *V = M[K];
    benchmark::DoNotOptimize(V);
    return V != nullptr;
  }

  bool BenchInsert(KeyT K, ValueT V) {
    auto Result = M.insert(K, V);
    benchmark::DoNotOptimize(Result.isInserted());
    return Result.isInserted();
  }
};

template <typename KT, typename VT, typename HasherT>
struct MapWrapper<absl::flat_hash_map<KT, VT, HasherT>> {
  using MapT = absl::flat_hash_map<KT, VT, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  bool BenchLookup(KeyT K) {
    auto It = M.find(K);
    benchmark::DoNotOptimize(It);
    return It != M.end();
  }

  bool BenchInsert(KeyT K, ValueT V) {
    auto Result = M.insert({K, V});
    benchmark::DoNotOptimize(Result.second);
    return Result.second;
  }
};

template <typename KT, typename VT, typename HasherT>
struct MapWrapper<llvm::DenseMap<KT, VT, HasherT>> {
  using MapT = llvm::DenseMap<KT, VT, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  bool BenchLookup(KeyT K) {
    auto It = M.find(K);
    benchmark::DoNotOptimize(It);
    return It != M.end();
  }

  bool BenchInsert(KeyT K, ValueT V) {
    auto Result = M.insert({K, V});
    benchmark::DoNotOptimize(Result.second);
    return Result.second;
  }
};

template <typename KT, typename VT, unsigned SmallSize, typename HasherT>
struct MapWrapper<llvm::SmallDenseMap<KT, VT, SmallSize, HasherT>> {
  using MapT = llvm::SmallDenseMap<KT, VT, SmallSize, HasherT>;
  using KeyT = KT;
  using ValueT = VT;

  MapT M;

  bool BenchLookup(KeyT K) {
    auto It = M.find(K);
    benchmark::DoNotOptimize(It);
    return It != M.end();
  }

  bool BenchInsert(KeyT K, ValueT V) {
    auto Result = M.insert({K, V});
    benchmark::DoNotOptimize(Result.second);
    return Result.second;
  }
};

struct LLVMHash {
  template <typename T>
  size_t operator()(const T &Arg) const {
    using llvm::hash_value;
    return hash_value(Arg);
  }
};

struct LLVMHashingDenseMapInfo {
  // The following should hold, but it would require int to be complete:
  // static_assert(alignof(int) <= (1 << Log2MaxAlign),
  //               "DenseMap does not support pointer keys requiring more than "
  //               "Log2MaxAlign bits of alignment");
  static constexpr uintptr_t Log2MaxAlign = 12;

  static inline int* getEmptyKey() {
    uintptr_t Val = static_cast<uintptr_t>(-1);
    Val <<= Log2MaxAlign;
    return reinterpret_cast<int*>(Val);
  }

  static inline int* getTombstoneKey() {
    uintptr_t Val = static_cast<uintptr_t>(-2);
    Val <<= Log2MaxAlign;
    return reinterpret_cast<int*>(Val);
  }

  static unsigned getHashValue(const int *PtrVal) {
    using llvm::hash_value;
    return hash_value(PtrVal);
  }

  static bool isEqual(const int *LHS, const int *RHS) { return LHS == RHS; }
};


using KeyVectorT = llvm::SmallVector<std::unique_ptr<int>, 32>;

KeyVectorT BuildKeys(ssize_t Size, llvm::function_ref<void(int *)> Callback = [](int *) {}) {
  KeyVectorT Keys;
  for (ssize_t i : llvm::seq<ssize_t>(0, Size))
    Keys.emplace_back(new int(i));

  for (ssize_t i : llvm::seq<ssize_t>(0, Size))
    Callback(Keys[i].get());

  return Keys;
}

constexpr ssize_t NumShuffledKeys = 1024 * 64;

llvm::SmallVector<int *, 32> BuildShuffledKeys(const KeyVectorT &Keys) {
  std::random_device RDev;
  std::seed_seq Seed(
      {RDev(), RDev(), RDev(), RDev(), RDev(), RDev(), RDev(), RDev()});
  std::mt19937_64 RNG(Seed);
  std::uniform_int_distribution<int> D(0, Keys.size() - 1);

  llvm::SmallVector<int *, 32> ShuffledKeys;
  for (ssize_t i : llvm::seq<ssize_t>(0, NumShuffledKeys)) {
    (void)i;
    ssize_t RandomIdx = D(RNG);
    assert(RandomIdx < (ssize_t)Keys.size() && "Too large value!");
    ShuffledKeys.push_back(Keys[RandomIdx].get());
  }

  return ShuffledKeys;
}

template <typename MapT>
static void BM_MapLookupHitPtr(benchmark::State& S) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  MapWrapperT M;
  KeyVectorT Keys = BuildKeys(S.range(0), [&M](int *Key) { M.BenchInsert(Key, T()); });
  llvm::SmallVector<int *, 32> ShuffledKeys = BuildShuffledKeys(Keys);

  ssize_t i = 0;
  for (auto _ : S) {
    bool Result = M.BenchLookup(ShuffledKeys[i]);
    assert(Result && "Lookup must succeed!");
    (void)Result;
    i = (i + 1) & (NumShuffledKeys - 1);
  }
}
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, Map<int *, std::array<int, 1>>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, Map<int *, std::array<int, 2>>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, Map<int *, std::array<int, 4>>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, Map<int *, std::array<int, 64>>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);

BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, absl::flat_hash_map<int *, std::array<int, 1>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, absl::flat_hash_map<int *, std::array<int, 2>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, absl::flat_hash_map<int *, std::array<int, 4>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, absl::flat_hash_map<int *, std::array<int, 64>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);

BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, llvm::DenseMap<int *, std::array<int, 1>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, llvm::DenseMap<int *, std::array<int, 2>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, llvm::DenseMap<int *, std::array<int, 4>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupHitPtr, llvm::DenseMap<int *, std::array<int, 64>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);


template <typename MapT>
static void BM_MapLookupMissPtr(benchmark::State& S) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  MapWrapperT M;
  KeyVectorT Keys = BuildKeys(S.range(0), [&M](int *Key) { M.BenchInsert(Key, T()); });
  constexpr ssize_t NumOtherKeys = 1024 * 64;
  KeyVectorT OtherKeys = BuildKeys(NumOtherKeys);

  ssize_t i = 0;
  for (auto _ : S) {
    bool Result = M.BenchLookup(OtherKeys[i].get());
    assert(!Result && "Lookup must fail!");
    (void)Result;
    i = (i + 1) & (NumOtherKeys - 1);
  }
}
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, Map<int *, std::array<int, 1>>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, Map<int *, std::array<int, 2>>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, Map<int *, std::array<int, 4>>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, Map<int *, std::array<int, 64>>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);

BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, absl::flat_hash_map<int *, std::array<int, 1>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, absl::flat_hash_map<int *, std::array<int, 2>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, absl::flat_hash_map<int *, std::array<int, 4>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, absl::flat_hash_map<int *, std::array<int, 64>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);

BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, llvm::DenseMap<int *, std::array<int, 1>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, llvm::DenseMap<int *, std::array<int, 2>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, llvm::DenseMap<int *, std::array<int, 4>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);
BENCHMARK_TEMPLATE(BM_MapLookupMissPtr, llvm::DenseMap<int *, std::array<int, 64>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(4)->Arg(8)->Arg(16)->Arg(32)
    ->Range(1 << 6, 1 << 20);


template <typename MapT>
static void BM_MapInsertPtrSeq(benchmark::State& S) {
  using MapWrapperT = MapWrapper<MapT>;
  using T = typename MapWrapperT::ValueT;
  KeyVectorT Keys = BuildKeys(S.range(0));
  llvm::SmallVector<int *, 32> ShuffledKeys = BuildShuffledKeys(Keys);

  ssize_t i = 0;
  for (auto _ : S) {
    // First insert all the keys.
    MapWrapperT M;
    for (const auto &K : Keys) {
      bool Inserted = M.BenchInsert(K.get(), T());
      assert(Inserted && "Must be a successful insert!");
      (void)Inserted;
    }

    // Now insert a final random repeated key.
    bool Inserted = M.BenchInsert(ShuffledKeys[i], T());
    assert(!Inserted && "Must already be in the map!");
    (void)Inserted;

    // Rotate through the shuffled keys.
    i = (i + 1) & (NumShuffledKeys - 1);
  }
}
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 1>, 4>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 2>, 4>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 4>, 4>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 64>, 4>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);

BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 1>, 8>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 2>, 8>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 4>, 8>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 64>, 8>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);

BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 1>, 32>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 2>, 32>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 4>, 32>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, Map<int *, std::array<int, 64>, 32>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);

BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, absl::flat_hash_map<int *, std::array<int, 1>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, absl::flat_hash_map<int *, std::array<int, 2>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, absl::flat_hash_map<int *, std::array<int, 4>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, absl::flat_hash_map<int *, std::array<int, 64>, LLVMHash>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);

BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::DenseMap<int *, std::array<int, 1>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::DenseMap<int *, std::array<int, 2>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::DenseMap<int *, std::array<int, 4>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::DenseMap<int *, std::array<int, 64>, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);

BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 1>, 4, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 2>, 4, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 4>, 4, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 64>, 4, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);

BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 1>, 8, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 2>, 8, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 4>, 8, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 64>, 8, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);

BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 1>, 32, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 2>, 32, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 4>, 32, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);
BENCHMARK_TEMPLATE(BM_MapInsertPtrSeq, llvm::SmallDenseMap<int *, std::array<int, 64>, 32, LLVMHashingDenseMapInfo>)
    ->Arg(1)->Arg(2)->Arg(3)->Arg(4)->Arg(5)->Arg(8)->Arg(9)->Arg(16)->Arg(17)->Arg(32)->Arg(33)
    ->Range(1 << 6, 1 << 15);

}  // namespace
}  // namespace Carbon::Testing
