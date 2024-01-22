// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/raw_hashtable_benchmark_helpers.h"

#include <forward_list>

namespace Carbon::RawHashtable {

// Build up a large collection of random and unique string keys. This is
// actually a relatively expensive operation due to needing to build all the
// random string text. As a consequence, the initializer of this global is
// somewhat performance tuned to ensure benchmarks take an excessive amount of
// time to run or use an excessive amount of memory.
static std::vector<llvm::StringRef> raw_str_keys = [] {
  std::vector<llvm::StringRef> keys;
  absl::BitGen gen;

  // For benchmarking, we use short strings in a fixed distribution with
  // common characters. Real-world strings aren't uniform across ASCII or
  // Unicode, etc. And for *micro*-benchmarking we want to focus on the map
  // overhead with short, fast keys.
  constexpr ssize_t NumChars = 64;
  static_assert(llvm::isPowerOf2_64(NumChars));
  constexpr ssize_t NumCharsMask = NumChars - 1;
  constexpr ssize_t NumCharsShift = llvm::CTLog2<NumChars>();
  std::vector<char> characters = {'_', '-'};
  for (auto range :
       {llvm::seq_inclusive('a', 'z'), llvm::seq_inclusive('A', 'Z'),
        llvm::seq_inclusive('0', '9')}) {
    for (char c : range) {
      characters.push_back(c);
    }
  }
  CARBON_CHECK(characters.size() == NumChars);

  std::array length_buckets = {
      4, 4, 4, 4, 5, 5, 5, 5, 7, 7, 10, 10, 15, 25, 40, 80,
  };
  static_assert((MaxNumKeys % length_buckets.size()) == 0);
  CARBON_CHECK(std::is_sorted(length_buckets.begin(), length_buckets.end()));

  // Directly construct every 4-character string. This is a little memory
  // intense, but ends up being much cheaper by letting us reliably select a
  // unique 4-character sequence.
  constexpr ssize_t NumFourCharStrs = NumChars * NumChars * NumChars * NumChars;
  static_assert(llvm::isPowerOf2_64(NumFourCharStrs));
  llvm::OwningArrayRef<std::array<char, 4>> four_char_strs(NumFourCharStrs);
  for (auto [i, str] : llvm::enumerate(four_char_strs)) {
    str[0] = characters[i & NumCharsMask];
    i >>= NumCharsShift;
    str[1] = characters[i & NumCharsMask];
    i >>= NumCharsShift;
    str[2] = characters[i & NumCharsMask];
    i >>= NumCharsShift;
    CARBON_CHECK((i & ~NumCharsMask) == 0);
    str[3] = characters[i];
  }

  // Now shuffle the 4-character strings.
  std::shuffle(four_char_strs.begin(), four_char_strs.end(), gen);

  // For each distinct length bucket, we build a vector of raw keys.
  std::forward_list<llvm::SmallVector<const char*>> raw_keys_storage;
  // And a parallel array to the length buckets with the raw keys of that
  // length.
  std::array<llvm::SmallVector<const char*>*, length_buckets.size()>
      raw_keys_buckets;

  ssize_t prev_length = -1;
  for (auto [length_index, length] : llvm::enumerate(length_buckets)) {
    if (length == prev_length) {
      // We can detect repetitions as the lengths are required to be sorted.
      raw_keys_buckets[length_index] = raw_keys_buckets[length_index - 1];
      continue;
    }
    prev_length = length;

    llvm::SmallVector<const char*>& raw_keys = raw_keys_storage.emplace_front();
    raw_keys_buckets[length_index] = &raw_keys;
    CARBON_CHECK(length >= 4);

    // Select a random start for indexing our four character strings.
    ssize_t four_char_index = absl::Uniform<ssize_t>(gen, 0, NumFourCharStrs);

    // We want to compute all the keys of this length that we'll need.
    ssize_t key_count = (MaxNumKeys / length_buckets.size()) *
                        llvm::count(length_buckets, length);

    // Do a single memory allocation for all the keys of this length to avoid an
    // excessive number of small and fragmented allocations. This memory is
    // intentionally leaked as the keys are global and will themselves will
    // point into it.
    char* key_text = new char[key_count * length];

    // Reserve all the key space since we know how many we'll need.
    raw_keys.reserve(key_count);
    for ([[gnu::unused]] ssize_t i : llvm::seq<ssize_t>(0, key_count)) {
      for (ssize_t j = 0; j < (length - 4); ++j) {
        key_text[j] = characters[absl::Uniform<ssize_t>(gen, 0, NumChars)];
      }

      // Set the last four characters with this entry in the shuffled sequence.
      memcpy(key_text + length - 4, four_char_strs[four_char_index].data(), 4);
      // Step through the shuffled sequence. We start at a random position, so
      // we need to wrap around the end.
      four_char_index = (four_char_index + 1) & (NumFourCharStrs - 1);

      // And finally save the start pointer as one of our raw keys.
      raw_keys.push_back(key_text);
      key_text += length;
    }
  }

  // Now reserve our actual key vector.
  keys.reserve(MaxNumKeys);
  // And build each element.
  for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
    // We round robin through the length buckets to find a particular length.
    ssize_t bucket = i % length_buckets.size();
    ssize_t length = length_buckets[bucket];
    // We pop a raw key from the list of them associated with this bucket.
    const char* raw_key = raw_keys_buckets[bucket]->pop_back_val();
    // And build our key from that.
    keys.push_back(llvm::StringRef(raw_key, length));
  }
  // Check that in fact we popped every raw key into our main keys.
  for (const auto& keys : raw_keys_storage) {
    CARBON_CHECK(keys.empty());
  }
  return keys;
}();

auto BuildRawStrKeys() -> llvm::ArrayRef<llvm::StringRef> {
  return raw_str_keys;
}

static std::vector<int*> raw_ptr_keys = [] {
  std::vector<int*> keys;
  for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
    // We leak these pointers -- this is a static initializer executed once.
    keys.push_back(new int(i));
  }
  return keys;
}();

auto BuildRawPtrKeys() -> llvm::ArrayRef<int*> { return raw_ptr_keys; }

static std::vector<int> raw_int_keys = [] {
  std::vector<int> keys;
  for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
    keys.push_back(i + 1);
  }
  return keys;
}();

auto BuildRawIntKeys() -> llvm::ArrayRef<int> { return raw_int_keys; }

}  // namespace Carbon::RawHashtable
