// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <benchmark/benchmark.h>

#include <algorithm>

#include "absl/random/random.h"
#include "common/raw_hashtable_metadata_group.h"

namespace Carbon::RawHashtable {
namespace {

constexpr ssize_t BenchSize = 256;

struct BenchMetadata {
  llvm::MutableArrayRef<uint8_t> metadata;

  // For random byte metadata, also store the specific byte to search for in
  // each group here. For other kinds, this is empty.
  llvm::ArrayRef<uint8_t> bytes;
};

enum class BenchKind {
  Random,
  Empty,
  Deleted,
};

template <BenchKind Kind = BenchKind::Random>
static auto BuildBenchMetadata() -> llvm::ArrayRef<BenchMetadata> {
  static uint8_t metadata_storage[GroupSize][BenchSize * GroupSize];
  static uint8_t bytes_storage[GroupSize][BenchSize];
  static BenchMetadata bm_storage[GroupSize];
  absl::BitGen gen;
  for (auto [bm_index, bm] : llvm::enumerate(bm_storage)) {
    int match_count = bm_index + 1;

    for (ssize_t g_index : llvm::seq<ssize_t>(0, BenchSize)) {
      // Start by filling the group with random bytes.
      auto group_bytes = llvm::MutableArrayRef(
          &metadata_storage[bm_index][g_index * GroupSize], GroupSize);
      for (uint8_t& b : group_bytes) {
        b = absl::Uniform<uint8_t>(gen) | MetadataGroup::PresentMask;
      }

      // Now we need up to `match_count` random indices into the group where
      // we'll put a matching byte.
      std::array<ssize_t, GroupSize> group_indices;
      std::iota(group_indices.begin(), group_indices.end(), 0);
      std::shuffle(group_indices.begin(), group_indices.end(), gen);

      // Now cause the first match index to have the desired value.
      ssize_t match_index = *group_indices.begin();
      uint8_t& match_b = group_bytes[match_index];
      switch (Kind) {
        case BenchKind::Random: {
          // Already a random value, but we need to  ensure it isn't one that
          // repeats elsewhere in the group.
          while (llvm::count(group_bytes, match_b) > 1) {
            match_b = absl::Uniform<uint8_t>(gen) | MetadataGroup::PresentMask;
          }
          // Store this as the byte to search for in this group.
          bytes_storage[bm_index][g_index] = match_b;
          break;
        }
        case BenchKind::Empty: {
          match_b = MetadataGroup::Empty;
          break;
        }
        case BenchKind::Deleted: {
          match_b = MetadataGroup::Deleted;
          break;
        }
      }

      // Replicate the match byte in each of the other matching indices.
      for (ssize_t m_index : llvm::ArrayRef(group_indices)
                                 .drop_front()
                                 .take_front(match_count - 1)) {
        group_bytes[m_index] = match_b;
      }
    }

    // Now that the storage is set up, record these in our struct.
    bm.metadata = metadata_storage[bm_index];
    if constexpr (Kind == BenchKind::Random) {
      bm.bytes = bytes_storage[bm_index];
    }
  }
  return bm_storage;
}

template <BenchKind Kind>
const auto bench_metadata = BuildBenchMetadata<Kind>();

template <BenchKind Kind, bool Portable = false>
static void BM_LoadMatch(benchmark::State& s) {
  BenchMetadata bm = bench_metadata<Kind>[0];
  ssize_t nonce_data[GroupSize];
  for (ssize_t& n : nonce_data) {
    n = 1;
  }
  // Now hide the contents of `nonce_data`.
  benchmark::ClobberMemory();
  ssize_t i = 0;
  benchmark::DoNotOptimize(i);
#pragma clang loop unroll(disable)
  for (auto _ : s) {
    auto g = MetadataGroup::Load(bm.metadata.data(), i * GroupSize);
    MetadataGroup::MatchIndex matches;
    if constexpr (!Portable) {
      if constexpr (Kind == BenchKind::Empty) {
        matches = g.MatchEmpty();
      } else if constexpr (Kind == BenchKind::Deleted) {
        matches = g.MatchDeleted();
      } else {
        static_assert(Kind == BenchKind::Random);
        matches = static_cast<MetadataGroup::MatchIndex>(g.Match(bm.bytes[i]));
      }
    } else {
      if constexpr (Kind == BenchKind::Empty) {
        matches = g.PortableMatchEmpty();
      } else if constexpr (Kind == BenchKind::Deleted) {
        matches = g.PortableMatchDeleted();
      } else {
        static_assert(Kind == BenchKind::Random);
        matches = static_cast<MetadataGroup::MatchIndex>(
            g.PortableMatch(bm.bytes[i]));
      }
    }
    CARBON_CHECK(matches);
    i = (i + nonce_data[matches.index()]) & (BenchSize - 1);
  }
}
BENCHMARK(BM_LoadMatch<BenchKind::Random>);
BENCHMARK(BM_LoadMatch<BenchKind::Empty>);
BENCHMARK(BM_LoadMatch<BenchKind::Deleted>);
BENCHMARK(BM_LoadMatch<BenchKind::Random, /*Portable=*/true>);
BENCHMARK(BM_LoadMatch<BenchKind::Empty, /*Portable=*/true>);
BENCHMARK(BM_LoadMatch<BenchKind::Deleted, /*Portable=*/true>);

template <BenchKind Kind, ssize_t Steps>
static void BM_LoadMatchMissSteps(benchmark::State& s) {
  static_assert(Steps > 0);
  static_assert(Steps <= GroupSize);
  BenchMetadata bm = bench_metadata<Kind>[Steps - 1];
  ssize_t nonce_data[GroupSize];
  for (ssize_t& n : nonce_data) {
    n = 1;
  }
  // Now hide the contents of `nonce_data`.
  benchmark::ClobberMemory();
  ssize_t i = 0;
  benchmark::DoNotOptimize(i);
#pragma clang loop unroll(disable)
  for (auto _ : s) {
    auto g = MetadataGroup::Load(bm.metadata.data(), i * GroupSize);
    auto matched_range = g.Match(bm.bytes[i]);
    ssize_t indices = 0;
    for (ssize_t index : matched_range) {
      indices += nonce_data[index];
    }
    i = (i + (indices - Steps + 1)) & (BenchSize - 1);
  }
}
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 1>);
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 2>);
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 4>);
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 8>);
#if CARBON_USE_X86_SIMD_CONTROL_GROUP
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 12>);
BENCHMARK(BM_LoadMatchMissSteps<BenchKind::Random, 16>);
#endif

}  // namespace
}  // namespace Carbon::RawHashtable
