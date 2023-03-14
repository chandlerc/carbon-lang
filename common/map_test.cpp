// Part of the Carbon Language project, under the Apache License v2.0 with LLVM
// Exceptions. See /LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "common/map.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <initializer_list>
#include <map>
#include <tuple>
#include <vector>

namespace Carbon::Testing {
namespace {

using ::testing::Pair;
using ::testing::UnorderedElementsAreArray;

template <typename MapT, typename MatcherRangeT>
void expectUnorderedElementsAre(MapT&& m, MatcherRangeT element_matchers) {
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
void expectUnorderedElementsAre(
    MapT&& m, std::initializer_list<MatcherT> element_matchers) {
  std::vector<MatcherT> element_matchers_storage = element_matchers;
  expectUnorderedElementsAre(m, element_matchers_storage);
}

TEST(MapTest, Basic) {
  Map<int, int, 4> m;
  using BaseT = MapBase<int, int>;
  using ViewT = MapView<int, int>;

  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);
  EXPECT_TRUE(m.Insert(1, 100).is_inserted());
  ASSERT_TRUE(m.Contains(1));
  ASSERT_TRUE(ViewT(m).Contains(1));
  auto result = m.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  EXPECT_EQ(100, result.value());
  EXPECT_EQ(100, *m[1]);
  result = ViewT(m).Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  EXPECT_EQ(100, result.value());
  EXPECT_EQ(100, *ViewT(m)[1]);
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
  expectUnorderedElementsAre(m, {Pair(1, 101)});

  // Fill up the small buffer but don't overflow it.
  for (int i : llvm::seq(2, 5)) {
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted()) << "Key: " << i;
  }
  for (int i : llvm::seq(1, 5)) {
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
  }
  EXPECT_FALSE(m.Contains(5));

  // Verify all the elements.
  expectUnorderedElementsAre(
      m, {Pair(1, 101), Pair(2, 201), Pair(3, 301), Pair(4, 401)});
  expectUnorderedElementsAre(
      *static_cast<BaseT*>(&m),
      {Pair(1, 101), Pair(2, 201), Pair(3, 301), Pair(4, 401)});
  expectUnorderedElementsAre(
      ViewT(m), {Pair(1, 101), Pair(2, 201), Pair(3, 301), Pair(4, 401)});

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
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted()) << "Key: " << i;
  }
  for (int i : llvm::seq(1, 14)) {
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i < 5), *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i < 5), *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i < 5), *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *m[i]) << "Key: " << i;
  }
  EXPECT_FALSE(m.Contains(42));

  // Verify all the elements by walking the entire map.
  expectUnorderedElementsAre(
      m, {Pair(1, 102), Pair(2, 202), Pair(3, 302), Pair(4, 402), Pair(5, 502),
          Pair(6, 602), Pair(7, 702), Pair(8, 802), Pair(9, 902),
          Pair(10, 1002), Pair(11, 1102), Pair(12, 1202), Pair(13, 1302)});

  // Now fill up several more control groups.
  for (int i : llvm::seq(14, 100)) {
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted()) << "Key: " << i;
  }
  for (int i : llvm::seq(1, 100)) {
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2 * (int)(i < 14), *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2 * (int)(i < 14), *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2 * (int)(i < 14), *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 3).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 3, *m[i]) << "Key: " << i;
  }
  EXPECT_FALSE(m.Contains(420));

  // Check walking the entire container.
  {
    std::vector<std::pair<int, int>> elements;
    for (int i : llvm::seq(1, 100)) {
      elements.push_back({i, i * 100 + 3});
    }
    expectUnorderedElementsAre(m, elements);
  }

  // Clear back to empty.
  m.Clear();
  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150)) {
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted()) << "Key: " << i;
  }
  for (int i : llvm::seq(50, 150)) {
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100, *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

  {
    std::vector<std::pair<int, int>> elements;
    for (int i : llvm::seq(50, 150)) {
      elements.push_back({i, i * 100 + 1});
    }
    expectUnorderedElementsAre(m, elements);
  }

  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Contains(73));
  EXPECT_TRUE(m.Erase(73));
  EXPECT_FALSE(m.Contains(73));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(m.Erase(i));
    EXPECT_FALSE(m.Contains(i));
  }
  for (int i : llvm::seq(50, 150)) {
    if (i == 73 || (i >= 102 && i < 136)) {
      continue;
    }
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *m[i]) << "Key: " << i;
  }
  EXPECT_TRUE(m.Insert(73, 73 * 100 + 3).is_inserted());
  EXPECT_EQ(73 * 100 + 3, *m[73]);

  {
    std::vector<std::pair<int, int>> elements;
    for (int i : llvm::seq(50, 150)) {
      if (i < 102 || i >= 136) {
        elements.push_back({i, i * 100 + 2 + (i == 73)});
      }
    }
    expectUnorderedElementsAre(m, elements);
  }

  // Reset back to empty and small.
  m.Reset();
  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);

  // Refill but with both overlapping and different values, now triggering
  // growth too. Also, use update instead of insert.
  for (int i : llvm::seq(75, 175)) {
    EXPECT_TRUE(m.Update(i, i * 100).is_inserted()) << "Key: " << i;
  }
  for (int i : llvm::seq(75, 175)) {
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100, *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

  {
    std::vector<std::pair<int, int>> elements;
    for (int i : llvm::seq(75, 175)) {
      elements.push_back({i, i * 100 + 1});
    }
    expectUnorderedElementsAre(ViewT(m), elements);
  }

  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Contains(93));
  EXPECT_TRUE(m.Erase(93));
  EXPECT_FALSE(m.Contains(93));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(m.Erase(i));
    EXPECT_FALSE(m.Contains(i));
  }
  for (int i : llvm::seq(75, 175)) {
    if (i == 93 || (i >= 102 && i < 136)) {
      continue;
    }
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *m[i]) << "Key: " << i;
  }
  EXPECT_TRUE(m.Insert(93, 93 * 100 + 3).is_inserted());
  EXPECT_EQ(93 * 100 + 3, *m[93]);

  {
    std::vector<std::pair<int, int>> elements;
    for (int i : llvm::seq(75, 175)) {
      if (i < 102 || i >= 136) {
        elements.push_back({i, i * 100 + 2 + (i == 93)});
      }
    }
    expectUnorderedElementsAre(ViewT(m), elements);
  }
}

TEST(MapTest, FactoryAPI) {
  Map<int, int, 4> m;
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

TEST(MapTest, BasicRef) {
  Map<int, int, 4> real_m;
  using ViewT = MapView<int, int>;
  using BaseT = MapBase<int, int>;

  BaseT& m = real_m;

  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);
  EXPECT_TRUE(m.Insert(1, 100).is_inserted());
  ASSERT_TRUE(m.Contains(1));
  ASSERT_TRUE(ViewT(m).Contains(1));
  auto result = m.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  EXPECT_EQ(100, result.value());
  EXPECT_EQ(100, *m[1]);
  result = ViewT(m).Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  EXPECT_EQ(100, result.value());
  EXPECT_EQ(100, *ViewT(m)[1]);
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

  // Now fill it up.
  for (int i : llvm::seq(2, 100)) {
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted()) << "Key: " << i;
  }
  for (int i : llvm::seq(1, 100)) {
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 3).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 3, *m[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(real_m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 3, *real_m[i]) << "Key: " << i;
  }
  EXPECT_FALSE(m.Contains(420));

  // Clear back to empty.
  m.Clear();
  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150)) {
    EXPECT_TRUE(m.Insert(i, i * 100).is_inserted()) << "Key: " << i;
  }
  for (int i : llvm::seq(50, 150)) {
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100, *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(real_m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *real_m[i]) << "Key: " << i;
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Contains(73));
  EXPECT_TRUE(m.Erase(73));
  EXPECT_FALSE(m.Contains(73));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(m.Erase(i));
    EXPECT_FALSE(m.Contains(i));
  }
  for (int i : llvm::seq(50, 150)) {
    if (i == 73 || (i >= 102 && i < 136)) {
      continue;
    }
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *m[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(real_m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *real_m[i]) << "Key: " << i;
  }
  EXPECT_TRUE(m.Insert(73, 73 * 100 + 3).is_inserted());
  EXPECT_EQ(73 * 100 + 3, *m[73]);

  // Reset back to empty and small.
  real_m.Reset();
  EXPECT_FALSE(m.Contains(42));
  EXPECT_EQ(nullptr, m[42]);

  // Refill but with both overlapping and different values, now triggering
  // growth too. Also, use update instead of insert.
  for (int i : llvm::seq(75, 175)) {
    EXPECT_TRUE(m.Update(i, i * 100).is_inserted()) << "Key: " << i;
  }
  for (int i : llvm::seq(75, 175)) {
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100, *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 1).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(real_m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *real_m[i]) << "Key: " << i;
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Contains(93));
  EXPECT_TRUE(m.Erase(93));
  EXPECT_FALSE(m.Contains(93));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(m.Erase(i));
    EXPECT_FALSE(m.Contains(i));
  }
  for (int i : llvm::seq(75, 175)) {
    if (i == 93 || (i >= 102 && i < 136)) {
      continue;
    }
    ASSERT_TRUE(m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(m).Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *ViewT(m)[i]) << "Key: " << i;
    EXPECT_FALSE(m.Insert(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *m[i]) << "Key: " << i;
    EXPECT_FALSE(m.Update(i, i * 100 + 2).is_inserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *m[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(real_m.Contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *real_m[i]) << "Key: " << i;
  }
  EXPECT_TRUE(m.Insert(93, 93 * 100 + 3).is_inserted());
  EXPECT_EQ(93 * 100 + 3, *m[93]);
}

}  // namespace
}  // namespace Carbon::Testing
