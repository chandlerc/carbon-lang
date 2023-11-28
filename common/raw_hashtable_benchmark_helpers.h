// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_
#define CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_

#include <benchmark/benchmark.h>
#include <sys/types.h>

#include "absl/random/random.h"
#include "common/check.h"
#include "common/hashing.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::RawHashtable {

// We want to support benchmarking with 1M keys plus up to 1k "other" keys (for
// misses).
constexpr ssize_t NumOtherKeys = 1 << 10;
constexpr ssize_t MaxNumKeys = (1 << 20) + NumOtherKeys;

auto BuildStrKeys(ssize_t size) -> llvm::ArrayRef<llvm::StringRef>;
auto BuildPtrKeys(ssize_t size) -> llvm::ArrayRef<int*>;
auto BuildIntKeys(ssize_t size) -> llvm::ArrayRef<int>;

template <typename T>
auto BuildKeys(ssize_t size) -> llvm::ArrayRef<T> {
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

inline auto OneOpSizeArgs(benchmark::internal::Benchmark* b) -> void {
  b->DenseRange(1, 8, 1);
  b->DenseRange(12, 16, 4);
  b->DenseRange(24, 64, 8);
  b->Range(1 << 7, 1 << 20);
}

inline auto OpSeqSizeArgs(benchmark::internal::Benchmark* b) -> void {
  b->DenseRange(1, 4, 1);
  b->Arg(8);
  b->Arg(16);
  b->Arg(32);
  b->Arg(64);
  b->Arg(128);
  b->Range(1 << 8, 1 << 18);
}

// Provide some Dense{Map,Set}Info viable implementations for the key types
// using Carbon's hashing framework. These let us benchmark the data structure
// alone rather than the combination of data structure and hashing routine.
//
// We only provide these for benchmarking -- they are *not* necessarily suitable
// for broader use. The Carbon hashing infrastructure has only been evaluated in
// the context of its specific hashtable design.
template <typename T>
struct CarbonHashingDenseInfo;

template <>
struct CarbonHashingDenseInfo<int> {
  static inline auto getEmptyKey() -> int { return -1; }
  static inline auto getTombstoneKey() -> int { return -2; }
  static auto getHashValue(const int val) -> unsigned {
    return static_cast<uint64_t>(HashValue(val));
  }
  static auto isEqual(const int lhs, const int rhs) -> bool {
    return lhs == rhs;
  }
};

template <typename T>
struct CarbonHashingDenseInfo<T*> {
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

  static auto isEqual(const T* lhs, const T* rhs) -> bool { return lhs == rhs; }
};

template <>
struct CarbonHashingDenseInfo<llvm::StringRef> {
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
  static auto isEqual(llvm::StringRef lhs, llvm::StringRef rhs) -> bool {
    if (rhs.data() == getEmptyKey().data()) {
      return lhs.data() == getEmptyKey().data();
    }
    if (rhs.data() == getTombstoneKey().data()) {
      return lhs.data() == getTombstoneKey().data();
    }
    return lhs == rhs;
  }
};

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_
