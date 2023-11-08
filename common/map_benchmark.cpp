// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <type_traits>

#include "absl/container/flat_hash_map.h"
#include "absl/random/random.h"
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

template <typename T>
struct CarbonHashingDenseInfo;

template <> struct CarbonHashingDenseInfo<int> {
  static inline auto getEmptyKey() -> int { return -1; }
  static inline auto getTombstoneKey() -> int { return -2; }
  static auto getHashValue(const int val) -> unsigned {
    return static_cast<uint64_t>(HashValue(val));
  }
  static auto isEqual(const int lhs, const int rhs) -> bool {
    return lhs == rhs;
  }
};

template <typename T> struct CarbonHashingDenseInfo<T*> {
  static constexpr uintptr_t Log2MaxAlign = 12;

  static inline auto getEmptyKey() -> T* {
    auto val = static_cast<uintptr_t>(-1);
    val <<= Log2MaxAlign;
    // NOLINTNEXTLINE(performance-no-int-to-ptr): This is required by the API.
    return reinterpret_cast<int*>(val);
  }

  static inline auto getTombstoneKey() -> T* {
    auto val = static_cast<uintptr_t>(-2);
    val <<= Log2MaxAlign;
    // NOLINTNEXTLINE(performance-no-int-to-ptr): This is required by the API.
    return reinterpret_cast<int*>(val);
  }

  static auto getHashValue(const T* ptr_val) -> unsigned {
    return static_cast<uint64_t>(HashValue(ptr_val));
  }

  static auto isEqual(const T* lhs, const T* rhs) -> bool {
    return lhs == rhs;
  }
};

template <> struct CarbonHashingDenseInfo<llvm::StringRef> {
  static auto getEmptyKey() -> llvm::StringRef {
    return llvm::StringRef(
        // NOLINTNEXTLINE(performance-no-int-to-ptr): Required by the API.
        reinterpret_cast<const char*>(~static_cast<uintptr_t>(0)), 0);
  }

  static auto getTombstoneKey() -> llvm::StringRef {
    return llvm::StringRef(
        // NOLINTNEXTLINE(performance-no-int-to-ptr): Required by the API.
        reinterpret_cast<const char*>(~static_cast<uintptr_t>(1)), 0);
  }
  static auto getHashValue(llvm::StringRef val) -> unsigned {
    return static_cast<uint64_t>(HashValue(val));
  }
  static bool isEqual(llvm::StringRef lhs, llvm::StringRef rhs) {
    if (rhs.data() == getEmptyKey().data()) {
      return lhs.data() == getEmptyKey().data();
    }
    if (rhs.data() == getTombstoneKey().data()) {
      return lhs.data() == getTombstoneKey().data();
    }
    return lhs == rhs;
  }
};

// We want to support benchmarking with 1M keys plus up to 1k "other" keys (for
// misses).
constexpr ssize_t NumOtherKeys = 1 << 10;
constexpr ssize_t MaxNumKeys = (1 << 20) + NumOtherKeys;

[[clang::noinline]] auto BuildStrKeys(ssize_t size)
    -> llvm::ArrayRef<llvm::StringRef> {
  static std::vector<llvm::StringRef> keys = [] {
    std::vector<llvm::StringRef> keys;

    // For benchmarking, we use short strings in a fixed distribution with
    // common characters. Real-world strings aren't uniform across ASCII or
    // Unicode, etc. And for *micro*-benchmarking we want to focus on the map
    // overhead with short, fast keys.
    std::vector<char> characters = {' ', '_', '-', '\n', '\t'};
    for (auto range :
         {llvm::seq_inclusive('a', 'z'), llvm::seq_inclusive('A', 'Z'),
          llvm::seq_inclusive('0', '9')}) {
      for (char c : range) {
        characters.push_back(c);
      }
    }
    ssize_t length_buckets[] = {
        4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 10, 10, 15, 25, 40, 80,
    };

    absl::BitGen gen;
    keys.reserve(MaxNumKeys);
    for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
      // We allocate and leak a string for each key. This is fine as we're a
      // static initializer only.
      std::string& s = *new std::string();

      ssize_t bucket = i % 16;
      ssize_t length = length_buckets[bucket];
      for ([[maybe_unused]] ssize_t j : llvm::seq<ssize_t>(0, length)) {
        s.push_back(
            characters[absl::Uniform<ssize_t>(gen, 0, characters.size())]);
      }
      keys.push_back(s);
    }
    return keys;
  }();
  return llvm::ArrayRef(keys).slice(0, size);
}

[[clang::noinline]] auto BuildPtrKeys(ssize_t size) -> llvm::ArrayRef<int*> {
  static std::vector<int*> keys = [] {
    std::vector<int*> keys;
    for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
      // We leak these pointers -- this is a static initializer executed once.
      keys.push_back(new int(i));
    }
    return keys;
  }();
  return llvm::ArrayRef(keys).slice(0, size);
}

[[clang::noinline]] auto BuildIntKeys(ssize_t size) -> llvm::ArrayRef<int> {
  static std::vector<int> keys = [] {
    std::vector<int> keys;
    for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
      keys.push_back(i);
    }
    return keys;
  }();
  return llvm::ArrayRef(keys).slice(0, size);
}

template <typename T> auto BuildKeys(ssize_t size) -> llvm::ArrayRef<T> {
  CARBON_CHECK(size <= MaxNumKeys);
  if constexpr (std::is_same_v<T, llvm::StringRef>) {
    return BuildStrKeys(size);
  } else if constexpr (std::is_pointer_v<T>) {
    return BuildPtrKeys(size);
  } else {
    return BuildIntKeys(size);
  }
}

// 64k shuffled keys to avoid any easily detectable pattern.
constexpr ssize_t NumShuffledKeys = 64 << 10;

template <typename T>
[[clang::noinline]] auto BuildShuffledKeys(llvm::ArrayRef<T> keys)
    -> llvm::SmallVector<T> {
  llvm::SmallVector<T> shuffled_keys;
  for ([[maybe_unused]] ssize_t i : llvm::seq<ssize_t>(0, NumShuffledKeys)) {
    shuffled_keys.push_back(keys[i % keys.size()]);
  }
  std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), absl::BitGen());
  return shuffled_keys;
}

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
    benchmark::DoNotOptimize(inserted);
    i = (i + static_cast<ssize_t>(inserted)) & (NumShuffledKeys - 1);
  }
}
MAP_BENCHMARK_ONE_OP(BM_MapEraseUpdateHit);

}  // namespace
}  // namespace Carbon::Testing
