// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/set.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <vector>

namespace Carbon {
namespace {

using ::testing::UnorderedElementsAreArray;

template <typename SetT, typename MatcherRangeT>
void ExpectSetElementsAre(SetT&& s, MatcherRangeT element_matchers) {
  // Now collect the elements into a container.
  using KeyT = typename std::remove_reference<SetT>::type::KeyT;
  std::vector<KeyT> entries;
  s.ForEach([&entries](KeyT& k) { entries.push_back(k); });

  // Use the GoogleMock unordered container matcher to validate and show errors
  // on wrong elements.
  EXPECT_THAT(entries, UnorderedElementsAreArray(element_matchers));
}

// Allow directly using an initializer list.
template <typename SetT, typename MatcherT>
void ExpectSetElementsAre(SetT&& s,
                          std::initializer_list<MatcherT> element_matchers) {
  std::vector<MatcherT> element_matchers_storage = element_matchers;
  ExpectSetElementsAre(s, element_matchers_storage);
}

template <typename RangeT, typename... RangeTs>
auto MakeElements(RangeT&& range, RangeTs&&... ranges) {
  std::vector<typename RangeT::value_type> elements;
  auto add_range = [&elements](RangeT&& r) {
    for (const auto&& e : r) {
      elements.push_back(e);
    }
  };
  add_range(std::forward<RangeT>(range));
  (add_range(std::forward<RangeT>(ranges)), ...);

  return elements;
}

TEST(SetTest, Basic) {
  Set<int, 16> s;

  EXPECT_FALSE(s.Contains(42));
  EXPECT_TRUE(s.Insert(1).is_inserted());
  EXPECT_TRUE(s.Contains(1));
  auto result = s.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  auto i_result = s.Insert(1);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_TRUE(s.Contains(1));

  // Verify all the elements.
  ExpectSetElementsAre(s, {1});

  // Fill up the small buffer but don't overflow it.
  for (int i : llvm::seq(2, 5)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 5)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Contains(i));
    EXPECT_FALSE(s.Insert(i).is_inserted());
  }
  EXPECT_FALSE(s.Contains(5));

  // Verify all the elements.
  ExpectSetElementsAre(s, {1, 2, 3, 4});

  // Erase some entries from the small buffer.
  EXPECT_FALSE(s.Erase(42));
  EXPECT_TRUE(s.Erase(2));
  EXPECT_TRUE(s.Contains(1));
  EXPECT_FALSE(s.Contains(2));
  EXPECT_TRUE(s.Contains(3));
  EXPECT_TRUE(s.Contains(4));
  EXPECT_TRUE(s.Erase(1));
  EXPECT_FALSE(s.Contains(1));
  EXPECT_FALSE(s.Contains(2));
  EXPECT_TRUE(s.Contains(3));
  EXPECT_TRUE(s.Contains(4));
  EXPECT_TRUE(s.Erase(4));
  EXPECT_FALSE(s.Contains(1));
  EXPECT_FALSE(s.Contains(2));
  EXPECT_TRUE(s.Contains(3));
  EXPECT_FALSE(s.Contains(4));
  // Fill them back in.
  EXPECT_TRUE(s.Insert(1).is_inserted());
  EXPECT_TRUE(s.Insert(2).is_inserted());
  EXPECT_TRUE(s.Insert(4).is_inserted());
  EXPECT_TRUE(s.Contains(1));
  EXPECT_TRUE(s.Contains(2));
  EXPECT_TRUE(s.Contains(3));
  EXPECT_TRUE(s.Contains(4));

  // Now fill up the first control group.
  for (int i : llvm::seq(5, 14)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 14)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Contains(i));
    EXPECT_FALSE(s.Insert(i).is_inserted());
    EXPECT_TRUE(s.Contains(i));
  }
  EXPECT_FALSE(s.Contains(42));

  // Verify all the elements.
  ExpectSetElementsAre(s, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});

  // Now fill up several more control groups.
  for (int i : llvm::seq(14, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Insert(i).is_inserted());
    ASSERT_TRUE(s.Contains(1));
  }
  for (int i : llvm::seq(1, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Contains(i));
    EXPECT_FALSE(s.Insert(i).is_inserted());
    EXPECT_TRUE(s.Contains(i));
  }
  EXPECT_FALSE(s.Contains(420));
  ExpectSetElementsAre(s, MakeElements(llvm::seq(1, 100)));

  // Clear back to empty.
  s.Clear();
  EXPECT_FALSE(s.Contains(42));

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Contains(i));
    EXPECT_FALSE(s.Insert(i).is_inserted());
    EXPECT_TRUE(s.Contains(i));
  }
  EXPECT_FALSE(s.Contains(42));
  EXPECT_FALSE(s.Contains(420));
  ExpectSetElementsAre(s, MakeElements(llvm::seq(50, 150)));

  EXPECT_FALSE(s.Erase(42));
  EXPECT_TRUE(s.Contains(73));
  EXPECT_TRUE(s.Erase(73));
  EXPECT_FALSE(s.Contains(73));
  for (int i : llvm::seq(102, 136)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Contains(i));
    EXPECT_TRUE(s.Erase(i));
    EXPECT_FALSE(s.Contains(i));
  }
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    if (i == 73 || (i >= 102 && i < 136)) {
      continue;
    }
    EXPECT_TRUE(s.Contains(i));
    EXPECT_FALSE(s.Insert(i).is_inserted());
    EXPECT_TRUE(s.Contains(i));
  }
  EXPECT_TRUE(s.Insert(73).is_inserted());
  EXPECT_TRUE(s.Contains(73));
  ExpectSetElementsAre(s,
                       MakeElements(llvm::seq(50, 102), llvm::seq(136, 150)));

  // Reset back to empty and small.
  s.Reset();
  EXPECT_FALSE(s.Contains(42));

  // Refill but with both overlapping and different values, now triggering
  // growth too.
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(s.Contains(i));
    EXPECT_FALSE(s.Insert(i).is_inserted());
    EXPECT_TRUE(s.Contains(i));
  }
  EXPECT_FALSE(s.Contains(42));
  EXPECT_FALSE(s.Contains(420));
  ExpectSetElementsAre(s, MakeElements(llvm::seq(75, 175)));

  EXPECT_FALSE(s.Erase(42));
  EXPECT_TRUE(s.Contains(93));
  EXPECT_TRUE(s.Erase(93));
  EXPECT_FALSE(s.Contains(93));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(s.Contains(i));
    EXPECT_TRUE(s.Erase(i));
    EXPECT_FALSE(s.Contains(i));
  }
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    if (i == 93 || (i >= 102 && i < 136)) {
      continue;
    }
    EXPECT_TRUE(s.Contains(i));
    EXPECT_FALSE(s.Insert(i).is_inserted());
    EXPECT_TRUE(s.Contains(i));
  }
  EXPECT_TRUE(s.Insert(93).is_inserted());
  EXPECT_TRUE(s.Contains(93));
  ExpectSetElementsAre(s,
                       MakeElements(llvm::seq(75, 102), llvm::seq(136, 175)));
}

TEST(SetTest, FactoryAPI) {
  Set<int, 16> s;
  EXPECT_TRUE(s.Insert(1, [](int k, void* key_storage) {
                 return new (key_storage) int(k);
               }).is_inserted());
  ASSERT_TRUE(s.Contains(1));
  // Reinsertion doesn't invoke the callback.
  EXPECT_FALSE(s.Insert(1, [](int, void*) -> int* {
                  llvm_unreachable("Should never be called!");
                }).is_inserted());
}

}  // namespace
}  // namespace Carbon
