// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/raw_hashtable_benchmark_helpers.h"

#include <vector>

#include "common/set.h"

namespace Carbon::RawHashtable {

auto BuildStrKeys(ssize_t size) -> llvm::ArrayRef<llvm::StringRef> {
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
    Set<llvm::StringRef> key_set;
    keys.reserve(MaxNumKeys);
    for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
      // We allocate and leak a string for each key. This is fine as we're a
      // static initializer only.
      std::string& s = *new std::string();

      ssize_t bucket = i % 16;
      ssize_t length = length_buckets[bucket];
      bool inserted = false;
      do {
        s.clear();
        for ([[maybe_unused]] ssize_t j : llvm::seq<ssize_t>(0, length)) {
          s.push_back(
              characters[absl::Uniform<ssize_t>(gen, 0, characters.size())]);
        }
        inserted = key_set.Insert(llvm::StringRef(s)).is_inserted();
        // Keep generating strings until we get a unique one.
      } while (!inserted);
      keys.push_back(llvm::StringRef(s));
    }
    return keys;
  }();
  return llvm::ArrayRef(keys).slice(0, size);
}

auto BuildPtrKeys(ssize_t size) -> llvm::ArrayRef<int*> {
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

auto BuildIntKeys(ssize_t size) -> llvm::ArrayRef<int> {
  static std::vector<int> keys = [] {
    std::vector<int> keys;
    for (ssize_t i : llvm::seq<ssize_t>(0, MaxNumKeys)) {
      keys.push_back(i);
    }
    return keys;
  }();
  return llvm::ArrayRef(keys).slice(0, size);
}

}  // namespace Carbon::RawHashtable