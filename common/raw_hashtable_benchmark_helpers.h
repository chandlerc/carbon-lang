// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_
#define CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_

#include <benchmark/benchmark.h>
#include <sys/types.h>

#include <limits>
#include <map>
#include <vector>

#include "absl/base/no_destructor.h"
#include "absl/random/random.h"
#include "common/check.h"
#include "common/hashing.h"
#include "common/raw_hashtable.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace Carbon::RawHashtable {

// We want to support benchmarking with 1M keys plus up to 256 "other" keys (for
// misses).
constexpr ssize_t NumOtherKeys = 1 << 8;
constexpr ssize_t MaxNumKeys = (1 << 24) + NumOtherKeys;

auto BuildRawStrKeys() -> llvm::ArrayRef<llvm::StringRef>;
auto BuildRawPtrKeys() -> llvm::ArrayRef<int*>;
auto BuildRawIntKeys() -> llvm::ArrayRef<int>;

template <typename T>
auto BuildRawKeys() -> llvm::ArrayRef<T> {
  if constexpr (std::is_same_v<T, llvm::StringRef>) {
    return BuildRawStrKeys();
  } else if constexpr (std::is_pointer_v<T>) {
    return BuildRawPtrKeys();
  } else {
    return BuildRawIntKeys();
  }
}

template <typename T>
auto GetKeysImpl(ssize_t size) -> llvm::ArrayRef<T> {
  // The raw keys aren't shuffled and round-robin through the sizes. We want to
  // keep the distribution of sizes extremely consistent so that we can compare
  // between two runs, even for small sizes. So for a given size we always take
  // the leading sequence from the raw keys for that size and then shuffle the
  // keys in that sequence to end up with a random sequence of keys. We store
  // each of these shuffled sequences in a map to avoid repeatedly computing
  // these on each benchmark run.
  static absl::NoDestructor<std::map<ssize_t, std::vector<T>>>
      shuffled_keys_by_size;

  std::vector<T>& shuffled_keys = (*shuffled_keys_by_size)[size];
  if (shuffled_keys.empty()) {
    llvm::ArrayRef<T> raw_keys = BuildRawKeys<T>();
    shuffled_keys.assign(raw_keys.begin(), raw_keys.begin() + size);
    std::shuffle(shuffled_keys.begin(), shuffled_keys.end(), absl::BitGen());
  }
  CARBON_CHECK(static_cast<ssize_t>(shuffled_keys.size()) == size);

  return llvm::ArrayRef(shuffled_keys);
}

template <typename T>
auto GetMissKeysImpl() -> llvm::ArrayRef<T> {
  // The raw keys aren't shuffled and round-robin through the sizes. We want to
  // keep the distribution of sizes extremely consistent so that we can compare
  // between two runs, even for small sizes. So for a given size we always take
  // the leading sequence from the raw keys for that size and then shuffle the
  // keys in that sequence to end up with a random sequence of keys. We store
  // each of these shuffled sequences in a map to avoid repeatedly computing
  // these on each benchmark run.
  static absl::NoDestructor<std::vector<T>> miss_keys([] {
    std::vector<T> keys;
    llvm::ArrayRef<T> raw_keys = BuildRawKeys<T>().take_back(NumOtherKeys);
    keys.assign(raw_keys.begin(), raw_keys.end());
    std::shuffle(keys.begin(), keys.end(), absl::BitGen());
    return keys;
  }());

  return llvm::ArrayRef(*miss_keys);
}

template <typename T>
auto GetKeysAndMissKeys(ssize_t size)
    -> std::pair<llvm::ArrayRef<T>, llvm::ArrayRef<T>> {
  CARBON_CHECK(size <= MaxNumKeys);
  return {BuildRawKeys<T>().slice(0, size), GetMissKeysImpl<T>()};
}

template <typename T>
auto GetKeysAndHitKeys(ssize_t size, ssize_t lookup_keys_size)
    -> std::pair<llvm::ArrayRef<T>, llvm::ArrayRef<T>> {
  CARBON_CHECK(size <= MaxNumKeys);
  CARBON_CHECK(lookup_keys_size <= MaxNumKeys);

  // See if we can just use a run of the keys for the lookup keys. It's tempting
  // to just extract a subsequence of the shuffled keys, but that can lead to a
  // skewed distribution of sizes in the lookup keys. Instead, we want to use a
  // sub-prefix of the underlying keys. We can do this by just re-using the
  // caching keys implementation.
  if (lookup_keys_size < size) {
    return {BuildRawKeys<T>().slice(0, size), GetKeysImpl<T>(lookup_keys_size)};
  }

  // Otherwise we'll need to build and cache a custom sequence of lookup keys
  // expanded to have duplicates.
  static absl::NoDestructor<
      std::map<std::pair<ssize_t, ssize_t>, std::vector<T>>>
      expanded_lookup_keys;
  std::vector<T>& lookup_keys =
      (*expanded_lookup_keys)[{size, lookup_keys_size}];
  if (lookup_keys.empty()) {
    llvm::ArrayRef<T> raw_keys = BuildRawKeys<T>();
    lookup_keys.reserve(lookup_keys_size);
    for (ssize_t i : llvm::seq<ssize_t>(0, lookup_keys_size)) {
      lookup_keys.push_back(raw_keys[i % size]);
    }
    std::shuffle(lookup_keys.begin(), lookup_keys.end(), absl::BitGen());
  }
  CARBON_CHECK(static_cast<ssize_t>(lookup_keys.size()) == lookup_keys_size);

  return {BuildRawKeys<T>().slice(0, size), llvm::ArrayRef(lookup_keys)};
}

// Convert values used in hashtable benchmarking to a bool. This is used to form
// dependencies between values stored in the hashtable between benchmark
// iterations.
template <typename T>
auto ValueToBool(T value) -> bool {
  if constexpr (std::is_same_v<T, llvm::StringRef>) {
    return value.size() > 0;
  } else if constexpr (std::is_pointer_v<T>) {
    return value != nullptr;
  } else {
    // We want our keys to include `0` for integers, so use the largest value.
    return value != std::numeric_limits<T>::max();
  }
}

inline auto SizeArgs(benchmark::internal::Benchmark* b) -> void {
  // Benchmarks for "miss" operations only have one parameter -- the size of the
  // table. These benchmarks use a fixed 1k set of extra keys for each miss
  // operation.
  b->DenseRange(1, 4, 1);
  b->Arg(8);
  b->Arg(16);
  b->Arg(32);

  // For sizes >= 64 we first use the power of two which will have a low load
  // factor, and then target exactly at our max load factor.
  auto large_sizes = {64, 1 << 8, 1 << 12, 1 << 16, 1 << 20, 1 << 24};
  for (auto s : large_sizes) {
    b->Arg(s);
  }
  for (auto s : large_sizes) {
    b->Arg(s - (s / 8));
  }
}

inline auto HitArgs(benchmark::internal::Benchmark* b) -> void {
  // There are two parameters for benchmarks of "hit" operations. The first is
  // the size of the hashtable itself. The second is the size of a buffer of
  // random keys actually in the hashtable to use for the operations.
  //
  // For small sizes, we use a fixed 1k lookup key count. This is enough to
  // avoid patterns of queries training the branch predictor just from the keys
  // themselves, while small enough to avoid significant L1 cache pressure.
  b->ArgsProduct({benchmark::CreateDenseRange(1, 4, 1), {NumOtherKeys}});
  b->Args({8, NumOtherKeys});
  b->Args({16, NumOtherKeys});
  b->Args({32, NumOtherKeys});

  // For sizes >= 64 we first use the power of two which will have a low load
  // factor, and then target exactly at our max load factor.
  std::vector<ssize_t> large_sizes = {64,      1 << 8,  1 << 12,
                                      1 << 16, 1 << 20, 1 << 24};
  for (auto i : llvm::seq<int>(0, large_sizes.size())) {
    ssize_t s = large_sizes[i];
    large_sizes.push_back(s - (s / 8));
  }

  for (auto s : large_sizes) {
    b->Args({s, NumOtherKeys});

    // Once the sizes are more than 4x the 1k minimum lookup buffer size, also
    // include 50% and 100% lookup buffer sizes.
    if (s >= (4 << 10)) {
      b->Args({s, s / 4});
      b->Args({s, s / 2});
    }
  }
}

// Provide some Dense{Map,Set}Info viable implementations for the key types
// using Carbon's hashing framework. These let us benchmark the data structure
// alone rather than the combination of data structure and hashing routine.
//
// We only provide these for benchmarking -- they are *not* necessarily suitable
// for broader use. The Carbon hashing infrastructure has only been evaluated in
// the context of its specific hashtable design.
template <typename T>
struct CarbonHashDI;

template <>
struct CarbonHashDI<int> {
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
struct CarbonHashDI<T*> {
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
struct CarbonHashDI<llvm::StringRef> {
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

template <typename T>
auto DumpHashStatistics(llvm::ArrayRef<T> keys) -> void {
  if (keys.size() < GroupSize) {
    return;
  }

  // The hash table load factor is 7/8ths, so we want to add 1/7th of our
  // current size, subtract one, and pick the next power of two to get the power
  // of two where 7/8ths is greater than or equal to the incoming key size.
  ssize_t expected_size =
      llvm::NextPowerOf2(keys.size() + (keys.size() / 7) - 1);

  constexpr int GroupShift = llvm::CTLog2<GroupSize>();

  auto get_hash_index = [expected_size](auto x) -> ssize_t {
    return HashValue(x, ComputeSeed())
               .template ExtractIndexAndTag<7>(expected_size)
               .first >>
           GroupShift;
  };

  std::vector<std::vector<int>> grouped_key_indices(expected_size >>
                                                    GroupShift);
  for (auto [i, k] : llvm::enumerate(keys)) {
    ssize_t hash_index = get_hash_index(k);
    CARBON_CHECK(hash_index < (expected_size >> GroupShift)) << hash_index;
    grouped_key_indices[hash_index].push_back(i);
  }
  ssize_t max_group_index =
      std::max_element(grouped_key_indices.begin(), grouped_key_indices.end(),
                       [](const auto& lhs, const auto& rhs) {
                         return lhs.size() < rhs.size();
                       }) -
      grouped_key_indices.begin();

  // If the max number of collisions on the index is less than or equal to the
  // group size, there shouldn't be any necessary probing (outside of deletion)
  // and so this isn't interesting, skip printing.
  if (grouped_key_indices[max_group_index].size() <= GroupSize) {
    return;
  }

  llvm::errs() << "keys: " << keys.size()
               << "  groups: " << grouped_key_indices.size() << "\n"
               << "max group index: " << llvm::formatv("{0x8}", max_group_index)
               << "  collisions: "
               << grouped_key_indices[max_group_index].size() << "\n";

  for (auto i : llvm::ArrayRef(grouped_key_indices[max_group_index])
                    .take_front(2 * GroupSize)) {
    auto k = keys[i];
    auto hash = static_cast<uint64_t>(HashValue(k, ComputeSeed()));
    uint64_t salt = ComputeSeed();
    llvm::errs() << "  key: " << k
                 << "  salt: " << llvm::formatv("{0:x16}", salt)
                 << "  hash: " << llvm::formatv("{0:x16}", hash) << "\n";
  }
}

}  // namespace Carbon::RawHashtable

#endif  // CARBON_COMMON_RAW_HASHTABLE_BENCHMARK_HELPERS_H_
