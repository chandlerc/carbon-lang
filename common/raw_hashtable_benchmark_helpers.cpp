// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/raw_hashtable_benchmark_helpers.h"

namespace Carbon::RawHashtable {

static std::vector<llvm::StringRef> raw_str_keys = [] {
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
  std::set<llvm::StringRef> key_set;
  keys.reserve(MaxNumKeys);
  for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
    // We allocate and leak a string for each key. This is fine as we're a
    // static initializer only.
    std::string& s = *new std::string();

    ssize_t bucket = i % 16;
    ssize_t length = length_buckets[bucket];
    s.reserve(length);
    do {
      s.clear();
      for ([[maybe_unused]] ssize_t j : llvm::seq<ssize_t>(0, length)) {
        s.push_back(
            characters[absl::Uniform<ssize_t>(gen, 0, characters.size())]);
      }
      // Keep generating strings until we get a unique one.
    } while (!key_set.insert(llvm::StringRef(s)).second);
    keys.push_back(llvm::StringRef(s));
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
    keys.push_back(i);
  }
  return keys;
}();

auto BuildRawIntKeys() -> llvm::ArrayRef<int> { return raw_int_keys; }

}  // namespace Carbon::RawHashtable
