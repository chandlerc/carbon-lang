// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/map.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <type_traits>
#include <utility>
#include <vector>

namespace Carbon::Testing {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAreArray;

template <typename MapT, typename MatcherRangeT>
void ExpectMapElementsAre(MapT&& m, MatcherRangeT element_matchers) {
  // Now collect the elements into a container.
  using KeyT = typename std::remove_reference<MapT>::type::KeyT;
  using ValueT = typename std::remove_reference<MapT>::type::ValueT;
  std::vector<std::pair<KeyT, ValueT>> map_entries;
  m.ForEach([&map_entries](KeyT& k, ValueT& v) {
    map_entries.push_back({k, v});
  });

  // Use the GoogleMock unordered container matcher to validate and show errors
  // on wrong elements.
  EXPECT_THAT(map_entries, UnorderedElementsAreArray(element_matchers));
}

// Allow directly using an initializer list.
template <typename MapT, typename MatcherT>
void ExpectMapElementsAre(
    MapT&& m, std::initializer_list<MatcherT> element_matchers) {
  std::vector<MatcherT> element_matchers_storage = element_matchers;
  ExpectMapElementsAre(m, element_matchers_storage);
}

template <typename ValueCB, typename RangeT, typename... RangeTs>
auto MakeKeyValues(ValueCB value_cb, RangeT&& range, RangeTs&&... ranges) {
  using KeyT = typename RangeT::value_type;
  using ValueT = decltype(value_cb(std::declval<KeyT>()));
  std::vector<std::pair<KeyT, ValueT>> elements;
  auto add_range = [&](RangeT&& r) {
    for (const auto&& e : r) {
      elements.push_back({e, value_cb(e)});
    }
  };
  add_range(std::forward<RangeT>(range));
  (add_range(std::forward<RangeT>(ranges)), ...);

  return elements;
}

TEST(MapTest, Basic) {
  Map<int, int, 16> m;

  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);
  EXPECT_TRUE(m.Insert(1, 100).is_inserted());
  ASSERT_TRUE(m.Contains(1));
  auto result = m.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  EXPECT_EQ(100, result.value());
  EXPECT_EQ(100, *m[1]);
  // Reinsertion doesn't change the value.
  auto i_result = m.Insert(1, 101);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(100, i_result.value());
  EXPECT_EQ(100, *m[1]);
  // Update does change the value.
  i_result = m.Update(1, 101);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(101, i_result.value());
  EXPECT_EQ(101, *m[1]);

  // Verify all the elements.
  ExpectMapElementsAre(m, {Pair(1, 101)});

  // Fill up the small buffer but don't overflow it.
  for (int i : llvm::seq(2, 5)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(1, 5)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + (int)(i == 1), *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + (int)(i == 1), *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
  }
  EXPECT_FALSE(m.Contains(5));

  // Verify all the elements.
  ExpectMapElementsAre(
      m, {Pair(1, 101), Pair(2, 201), Pair(3, 301), Pair(4, 401)});

  // Erase some entries from the small buffer.
  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Erase(2));
  EXPECT_EQ(101, *m[1]);
  EXPECT_EQ(nullptr, m[2]);
  EXPECT_EQ(301, *m[3]);
  EXPECT_EQ(401, *m[4]);
  EXPECT_TRUE(m.Erase(1));
  EXPECT_EQ(nullptr, m[1]);
  EXPECT_EQ(nullptr, m[2]);
  EXPECT_EQ(301, *m[3]);
  EXPECT_EQ(401, *m[4]);
  EXPECT_TRUE(m.Erase(4));
  EXPECT_EQ(nullptr, m[1]);
  EXPECT_EQ(nullptr, m[2]);
  EXPECT_EQ(301, *m[3]);
  EXPECT_EQ(nullptr, m[4]);
  // Fill them back in, but with a different order and going back to the
  // original value.
  EXPECT_TRUE(m.Insert(1, 100).is_inserted());
  EXPECT_TRUE(m.Insert(2, 200).is_inserted());
  EXPECT_TRUE(m.Insert(4, 400).is_inserted());
  EXPECT_EQ(100, *m[1]);
  EXPECT_EQ(200, *m[2]);
  EXPECT_EQ(301, *m[3]);
  EXPECT_EQ(400, *m[4]);
  // Then update their values to match.
  EXPECT_FALSE(m.Update(1, 101).is_inserted());
  EXPECT_FALSE(m.Update(2, 201).is_inserted());
  EXPECT_FALSE(m.Update(4, 401).is_inserted());

  // Now fill up the first control group.
  for (int i : llvm::seq(5, 14)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(1, 14)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + (int)(i < 5), *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + (int)(i < 5), *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 2, *m[i]);
  }
  EXPECT_FALSE(m.Contains(42));

  // Verify all the elements by walking the entire map.
  ExpectMapElementsAre(
      m, {Pair(1, 102), Pair(2, 202), Pair(3, 302), Pair(4, 402), Pair(5, 502),
          Pair(6, 602), Pair(7, 702), Pair(8, 802), Pair(9, 902),
          Pair(10, 1002), Pair(11, 1102), Pair(12, 1202), Pair(13, 1302)});

  // Now fill up several more control groups.
  for (int i : llvm::seq(14, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(1, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + 2 * (int)(i < 14), *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 2 * (int)(i < 14), *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 3).is_inserted());
    EXPECT_EQ(i * 100 + 3, *m[i]);
  }
  EXPECT_FALSE(m.Contains(420));

  // Check walking the entire container.
  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 3; }, llvm::seq(1, 100)));

  // Clear back to empty.
  m.Clear();
  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150)) {
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100, *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100, *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 1; }, llvm::seq(50, 150)));

  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Contains(73));
  EXPECT_TRUE(m.Erase(73));
  EXPECT_FALSE(m.Contains(73));
  for (int i : llvm::seq(102, 136)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(m.Erase(i));
    EXPECT_FALSE(m.Contains(i));
  }
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    if (i == 73 || (i >= 102 && i < 136)) {
      continue;
    }
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + 1, *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 2, *m[i]);
  }
  EXPECT_TRUE(m.Insert(73, 73 * 100 + 3).is_inserted());
  EXPECT_EQ(73 * 100 + 3, *m[73]);

  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 2 + (k == 73); },
                       llvm::seq(50, 102), llvm::seq(136, 150)));

  // Reset back to empty and small.
  m.Reset();
  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);

  // Refill but with both overlapping and different values, now triggering
  // growth too. Also, use update instead of insert.
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Update(i, i * 100).is_inserted());
  }
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100, *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100, *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 1; }, llvm::seq(75, 175)));

  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Contains(93));
  EXPECT_TRUE(m.Erase(93));
  EXPECT_FALSE(m.Contains(93));
  for (int i : llvm::seq(102, 136)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(m.Erase(i));
    EXPECT_FALSE(m.Contains(i));
  }
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    if (i == 93 || (i >= 102 && i < 136)) {
      continue;
    }
    ASSERT_TRUE(m.Contains(i));
    EXPECT_EQ(i * 100 + 1, *m[i]);
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 1, *m[i]);
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted());
    EXPECT_EQ(i * 100 + 2, *m[i]);
  }
  EXPECT_TRUE(m.Insert(93, 93 * 100 + 3).is_inserted());
  EXPECT_EQ(93 * 100 + 3, *m[93]);

  ExpectMapElementsAre(
      m, MakeKeyValues([](int k) { return k * 100 + 2 + (k == 93); },
                       llvm::seq(75, 102), llvm::seq(136, 175)));
}

TEST(MapTest, FactoryAPI) {
  Map<int, int, 16> m;
  EXPECT_TRUE(m.Insert(1, [] { return 100; }).is_inserted());
  ASSERT_TRUE(m.Contains(1));
  EXPECT_EQ(100, *m[1]);
  // Reinsertion doesn't invoke the callback.
  EXPECT_FALSE(m.Insert(1, []() -> int {
                  llvm_unreachable("Should never be called!");
                }).is_inserted());
  // Update does invoke the callback.
  auto i_result = m.Update(1, [] { return 101; });
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_EQ(101, i_result.value());
  EXPECT_EQ(101, *m[1]);
}

}  // namespace
}  // namespace Carbon::Testing
