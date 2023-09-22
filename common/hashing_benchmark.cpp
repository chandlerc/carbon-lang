// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>
#include <cstddef>

#include "absl/hash/hash.h"
#include "absl/random/random.h"
#include "common/check.h"
#include "common/hashing.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/StringExtras.h"

namespace Carbon::Testing {
namespace {

// We want the benchmark working set to fit in the L1 cache where possible so
// that the benchmark focuses on the CPU-execution costs and not memory latency.
// For most CPUs we're going to care about, 16k will fit easily, and 32k will
// probably fit. But we also need to include sizes for string benchmarks. This
// targets 8k of entropy with each object up to 8k of size for a total of 16k.
constexpr int EntropySize = 8 << 10;
constexpr int EntropyObjSize = 8 << 10;

// This returns an array to random entropy with `EntropySize` bytes plus 1k. The
// goal is that clients can read `EntropySize` objects out of this pool by
// starting at different byte offsets.
static const llvm::ArrayRef<std::byte> EntropyBytes =
    []() -> llvm::ArrayRef<std::byte> {
  static llvm::SmallVector<std::byte> bytes;
  // Pad out the entropy for up to 1kb objects.
  bytes.resize(EntropySize + EntropyObjSize);
  absl::BitGen gen;
  for (std::byte& b : bytes) {
    b = static_cast<std::byte>(absl::Uniform<uint8_t>(gen));
  }
  return bytes;
}();

// Based on 16k of entropy above and an L1 cache size often up to 32k, keep this
// small at 8k or 1k 8-byte sizes.
constexpr int NumSizes = 1 << 10;

template <size_t MaxSize>
static const std::array<size_t, NumSizes> rand_sizes = []() {
  std::array<size_t, NumSizes> sizes;
  // Build an array with a completely deterministic set of sizes in the
  // range [0, MaxSize). We scale the steps in sizes to cover the range at least
  // 128 times, even if it means not covering all the sizes within that range.
  static_assert(NumSizes > 128);
  constexpr size_t Scale = std::max<size_t>(1, MaxSize / (NumSizes / 128));
  for (auto [i, size] : llvm::enumerate(sizes)) {
    size = (i * Scale) % MaxSize;
  }
  // Shuffle the sizes randomly so that there isn't any pattern of sizes
  // encountered and we get relatively realistic branch prediction behavior
  // when branching on the size. We use this approach rather than random
  // sizes to ensure we always have the same total size of data processed.
  std::shuffle(sizes.begin(), sizes.end(), absl::BitGen());
  return sizes;
}();

template <typename T>
struct RandValues {
  size_t bytes = 0;

  auto Get(ssize_t /*i*/, uint64_t x) -> T {
    static_assert(sizeof(T) <= EntropyObjSize);
    bytes += sizeof(T);
    T result;
    memcpy(&result, &EntropyBytes[x % EntropySize], sizeof(T));
    return result;
  }
};

template <typename T, typename U>
struct RandValues<std::pair<T, U>> {
  size_t bytes = 0;

  auto Get(ssize_t /*i*/, uint64_t x) -> std::pair<T, U> {
    static_assert(sizeof(std::pair<T, U>) <= EntropyObjSize);
    bytes += sizeof(std::pair<T, U>);
    T result0;
    U result1;
    memcpy(&result0, &EntropyBytes[x % EntropySize], sizeof(T));
    memcpy(&result1, &EntropyBytes[x % EntropySize] + sizeof(T), sizeof(U));
    return {result0, result1};
  }
};

template <size_t MaxSize>
struct RandStrings {
  static constexpr size_t NumSizes = 1024;

  size_t bytes = 0;

  auto Get(ssize_t i, uint64_t x) -> llvm::StringRef {
    // This has a small bias towards small numbers. Because max N is ~200 this
    // is very small and prefer to be very fast instead of absolutely accurate.
    // Also we pass MaxSize = 2^K+1 so that mod reduces to a bitand.
    size_t s = rand_sizes<MaxSize>[i % NumSizes];
    bytes += s;
    return llvm::StringRef(
        reinterpret_cast<const char*>(&EntropyBytes[x % EntropySize]), s);
  }
};

struct CarbonHasher {
  template <typename T>
  __attribute__((noinline)) auto operator()(const T& value) -> uint64_t {
    auto result = static_cast<uint64_t>(HashValue(value));
    return result;
  }
};

struct CarbonSeededHasher {
  HashCode seed;

  CarbonSeededHasher() {
    volatile char key;
    seed = HashValue(&key);
  }

  template <typename T>
  auto operator()(const T& value) -> uint64_t {
    return static_cast<uint64_t>(HashValue(value, seed));
  }
};

struct AbslHasher {
  template <typename T>
  auto operator()(const T& value) -> uint64_t {
    return absl::HashOf(value);
  }
};

struct AbslSeededHasher {
  uint64_t seed;

  AbslSeededHasher() {
    volatile char key;
    seed = absl::HashOf(&key);
  }

  template <typename T>
  auto operator()(const T& value) -> uint64_t {
    return absl::HashOf(value) ^ seed;
  }
};

struct LLVMHasher {
  template <typename T>
  auto operator()(const T& value) -> uint64_t {
    return llvm::hash_value(value);
  }
};

struct LLVMSeededHasher {
  uint64_t seed;

  LLVMSeededHasher() {
    volatile char key;
    seed = llvm::hash_value(&key);
  }

  template <typename T>
  auto operator()(const T& value) -> uint64_t {
    return llvm::hash_value(value) ^ seed;
  }
};

template <typename Values, typename Hasher>
void BM_LatencyHash(benchmark::State& state) {
  ssize_t i = 0;
  uint64_t x = 13;
  Values v;
  Hasher h;
  for (auto _ : state) {
    benchmark::DoNotOptimize(x = h(v.Get(i++, x)));
  }
  state.SetBytesProcessed(v.bytes);
}

#define LATENCY_VALUE_BENCHMARKS(...)                                     \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, CarbonHasher>);       \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, AbslHasher>);         \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, LLVMHasher>);         \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, CarbonSeededHasher>); \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, AbslSeededHasher>);   \
  BENCHMARK(BM_LatencyHash<RandValues<__VA_ARGS__>, LLVMSeededHasher>)
LATENCY_VALUE_BENCHMARKS(uint8_t);
LATENCY_VALUE_BENCHMARKS(uint16_t);
LATENCY_VALUE_BENCHMARKS(std::pair<uint8_t, uint8_t>);
LATENCY_VALUE_BENCHMARKS(uint32_t);
LATENCY_VALUE_BENCHMARKS(std::pair<uint16_t, uint16_t>);
LATENCY_VALUE_BENCHMARKS(uint64_t);
LATENCY_VALUE_BENCHMARKS(int*);
LATENCY_VALUE_BENCHMARKS(std::pair<uint32_t, uint32_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<uint64_t, uint32_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<uint32_t, uint64_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<int*, uint32_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<uint32_t, int*>);
LATENCY_VALUE_BENCHMARKS(__uint128_t);
LATENCY_VALUE_BENCHMARKS(std::pair<uint64_t, uint64_t>);
LATENCY_VALUE_BENCHMARKS(std::pair<int*, int*>);
LATENCY_VALUE_BENCHMARKS(std::pair<uint64_t, int*>);
LATENCY_VALUE_BENCHMARKS(std::pair<int*, uint64_t>);

#define LATENCY_STRING_BENCHMARKS(MaxSize)                             \
  BENCHMARK(BM_LatencyHash<RandStrings<MaxSize>, CarbonHasher>);       \
  BENCHMARK(BM_LatencyHash<RandStrings<MaxSize>, AbslHasher>);         \
  BENCHMARK(BM_LatencyHash<RandStrings<MaxSize>, LLVMHasher>);         \
  BENCHMARK(BM_LatencyHash<RandStrings<MaxSize>, CarbonSeededHasher>); \
  BENCHMARK(BM_LatencyHash<RandStrings<MaxSize>, AbslSeededHasher>);   \
  BENCHMARK(BM_LatencyHash<RandStrings<MaxSize>, LLVMSeededHasher>)

LATENCY_STRING_BENCHMARKS(8);
LATENCY_STRING_BENCHMARKS(16);
LATENCY_STRING_BENCHMARKS(32);
LATENCY_STRING_BENCHMARKS(64);
LATENCY_STRING_BENCHMARKS(256);
LATENCY_STRING_BENCHMARKS(1024);
LATENCY_STRING_BENCHMARKS(8192);

}  // namespace
}  // namespace Carbon::Testing
