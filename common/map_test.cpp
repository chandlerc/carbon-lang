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
void expectUnorderedElementsAre(
    MapT &&M, MatcherRangeT ElementMatchers) {
  // Now collect the elements into a container.
  using KeyT = typename std::remove_reference<MapT>::type::KeyT;
  using ValueT = typename std::remove_reference<MapT>::type::ValueT;
  std::vector<std::pair<KeyT, ValueT>> MapEntries;
  M.forEach([&MapEntries](KeyT &K, ValueT &V) {
    MapEntries.push_back({K, V});
  });

  // Use the GoogleMock unordered container matcher to validate and show errors
  // on wrong elements.
  EXPECT_THAT(MapEntries, UnorderedElementsAreArray(ElementMatchers));
}

// Allow directly using an initializer list.
template <typename MapT, typename MatcherT>
void expectUnorderedElementsAre(
    MapT &&M, std::initializer_list<MatcherT> ElementMatchers) {
  std::vector<MatcherT> ElementMatchersStorage = ElementMatchers;
  expectUnorderedElementsAre(M, ElementMatchersStorage);
}

TEST(MapTest, Basic) {
  Map<int, int, 4> M;
  using RefT = MapRef<int, int>;
  using ViewT = MapView<int, int>;

  EXPECT_FALSE(M.contains(42));
  EXPECT_EQ(nullptr, M[42]);
  EXPECT_TRUE(M.insert(1, 100).isInserted());
  ASSERT_TRUE(M.contains(1));
  ASSERT_TRUE(ViewT(M).contains(1));
  auto Result = M.lookup(1);
  EXPECT_TRUE(Result);
  EXPECT_EQ(1, Result.getKey());
  EXPECT_EQ(100, Result.getValue());
  EXPECT_EQ(100, *M[1]);
  Result = ViewT(M).lookup(1);
  EXPECT_TRUE(Result);
  EXPECT_EQ(1, Result.getKey());
  EXPECT_EQ(100, Result.getValue());
  EXPECT_EQ(100, *ViewT(M)[1]);
  // Reinsertion doesn't change the value.
  auto IResult = M.insert(1, 101);
  EXPECT_FALSE(IResult.isInserted());
  EXPECT_EQ(100, IResult.getValue());
  EXPECT_EQ(100, *M[1]);
  // Update does change the value.
  IResult = M.update(1, 101);
  EXPECT_FALSE(IResult.isInserted());
  EXPECT_EQ(101, IResult.getValue());
  EXPECT_EQ(101, *M[1]);

  // Verify all the elements.
  expectUnorderedElementsAre(M, {Pair(1, 101)});

  // Fill up the small buffer but don't overflow it.
  for (int i : llvm::seq(2, 5))
    EXPECT_TRUE(M.insert(i, i * 100).isInserted()) << "Key: " << i;
  for (int i : llvm::seq(1, 5)) {
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
  }
  EXPECT_FALSE(M.contains(5));

  // Verify all the elements.
  expectUnorderedElementsAre(
      M, {Pair(1, 101), Pair(2, 201), Pair(3, 301), Pair(4, 401)});
  expectUnorderedElementsAre(
      RefT(M), {Pair(1, 101), Pair(2, 201), Pair(3, 301), Pair(4, 401)});
  expectUnorderedElementsAre(
      ViewT(M), {Pair(1, 101), Pair(2, 201), Pair(3, 301), Pair(4, 401)});

  // Erase some entries from the small buffer.
  EXPECT_FALSE(M.erase(42));
  EXPECT_TRUE(M.erase(2));
  EXPECT_EQ(101, *M[1]);
  EXPECT_EQ(nullptr, M[2]);
  EXPECT_EQ(301, *M[3]);
  EXPECT_EQ(401, *M[4]);
  EXPECT_TRUE(M.erase(1));
  EXPECT_EQ(nullptr, M[1]);
  EXPECT_EQ(nullptr, M[2]);
  EXPECT_EQ(301, *M[3]);
  EXPECT_EQ(401, *M[4]);
  EXPECT_TRUE(M.erase(4));
  EXPECT_EQ(nullptr, M[1]);
  EXPECT_EQ(nullptr, M[2]);
  EXPECT_EQ(301, *M[3]);
  EXPECT_EQ(nullptr, M[4]);
  // Fill them back in, but with a different order and going back to the
  // original value.
  EXPECT_TRUE(M.insert(1, 100).isInserted());
  EXPECT_TRUE(M.insert(2, 200).isInserted());
  EXPECT_TRUE(M.insert(4, 400).isInserted());
  EXPECT_EQ(100, *M[1]);
  EXPECT_EQ(200, *M[2]);
  EXPECT_EQ(301, *M[3]);
  EXPECT_EQ(400, *M[4]);
  // Then update their values to match.
  EXPECT_FALSE(M.update(1, 101).isInserted());
  EXPECT_FALSE(M.update(2, 201).isInserted());
  EXPECT_FALSE(M.update(4, 401).isInserted());

  // Now fill up the first control group.
  for (int i : llvm::seq(5, 14))
    EXPECT_TRUE(M.insert(i, i * 100).isInserted()) << "Key: " << i;
  for (int i : llvm::seq(1, 14)) {
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i < 5), *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i < 5), *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i < 5), *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *M[i]) << "Key: " << i;
  }
  EXPECT_FALSE(M.contains(42));

  // Verify all the elements by walking the entire map.
  expectUnorderedElementsAre(
      M, {Pair(1, 102), Pair(2, 202), Pair(3, 302), Pair(4, 402), Pair(5, 502),
          Pair(6, 602), Pair(7, 702), Pair(8, 802), Pair(9, 902),
          Pair(10, 1002), Pair(11, 1102), Pair(12, 1202), Pair(13, 1302)});

  // Now fill up several more control groups.
  for (int i : llvm::seq(14, 100))
    EXPECT_TRUE(M.insert(i, i * 100).isInserted()) << "Key: " << i;
  for (int i : llvm::seq(1, 100)) {
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2 * (int)(i < 14), *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2 * (int)(i < 14), *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2 * (int)(i < 14), *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 3).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 3, *M[i]) << "Key: " << i;
  }
  EXPECT_FALSE(M.contains(420));

  // Check walking the entire container.
  {
    std::vector<std::pair<int, int>> Elements;
    for (int i : llvm::seq(1, 100))
      Elements.push_back({i, i * 100 + 3});
    expectUnorderedElementsAre(M, Elements);
  }

  // Clear back to empty.
  M.clear();
  EXPECT_FALSE(M.contains(42));
  EXPECT_EQ(nullptr, M[42]);

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150))
    EXPECT_TRUE(M.insert(i, i * 100).isInserted()) << "Key: " << i;
  for (int i : llvm::seq(50, 150)) {
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100, *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
  }
  EXPECT_FALSE(M.contains(42));
  EXPECT_FALSE(M.contains(420));

  {
    std::vector<std::pair<int, int>> Elements;
    for (int i : llvm::seq(50, 150))
      Elements.push_back({i, i * 100 + 1});
    expectUnorderedElementsAre(M, Elements);
  }

  EXPECT_FALSE(M.erase(42));
  EXPECT_TRUE(M.contains(73));
  EXPECT_TRUE(M.erase(73));
  EXPECT_FALSE(M.contains(73));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(M.contains(i));
    EXPECT_TRUE(M.erase(i));
    EXPECT_FALSE(M.contains(i));
  }
  for (int i : llvm::seq(50, 150)) {
    if (i == 73 || (i >= 102 && i < 136))
      continue;
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *M[i]) << "Key: " << i;
  }
  EXPECT_TRUE(M.insert(73, 73 * 100 + 3).isInserted());
  EXPECT_EQ(73 * 100 + 3, *M[73]);

  {
    std::vector<std::pair<int, int>> Elements;
    for (int i : llvm::seq(50, 150))
      if (i < 102 || i >= 136)
        Elements.push_back({i, i * 100 + 2 + (i == 73)});
    expectUnorderedElementsAre(M, Elements);
  }

  // Reset back to empty and small.
  M.reset();
  EXPECT_FALSE(M.contains(42));
  EXPECT_EQ(nullptr, M[42]);

  // Refill but with both overlapping and different values, now triggering
  // growth too. Also, use update instead of insert.
  for (int i : llvm::seq(75, 175))
    EXPECT_TRUE(M.update(i, i * 100).isInserted()) << "Key: " << i;
  for (int i : llvm::seq(75, 175)) {
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100, *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
  }
  EXPECT_FALSE(M.contains(42));
  EXPECT_FALSE(M.contains(420));

  {
    std::vector<std::pair<int, int>> Elements;
    for (int i : llvm::seq(75, 175))
      Elements.push_back({i, i * 100 + 1});
    expectUnorderedElementsAre(ViewT(M), Elements);
  }

  EXPECT_FALSE(M.erase(42));
  EXPECT_TRUE(M.contains(93));
  EXPECT_TRUE(M.erase(93));
  EXPECT_FALSE(M.contains(93));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(M.contains(i));
    EXPECT_TRUE(M.erase(i));
    EXPECT_FALSE(M.contains(i));
  }
  for (int i : llvm::seq(75, 175)) {
    if (i == 93 || (i >= 102 && i < 136))
      continue;
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *M[i]) << "Key: " << i;
  }
  EXPECT_TRUE(M.insert(93, 93 * 100 + 3).isInserted());
  EXPECT_EQ(93 * 100 + 3, *M[93]);

  {
    std::vector<std::pair<int, int>> Elements;
    for (int i : llvm::seq(75, 175))
      if (i < 102 || i >= 136)
        Elements.push_back({i, i * 100 + 2 + (i == 93)});
    expectUnorderedElementsAre(ViewT(M), Elements);
  }
}

TEST(MapTest, FactoryAPI) {
  Map<int, int, 4> M;
  EXPECT_TRUE(M.insert(1, [] { return 100; }).isInserted());
  ASSERT_TRUE(M.contains(1));
  EXPECT_EQ(100, *M[1]);
  // Reinsertion doesn't invoke the callback.
  EXPECT_FALSE(M.insert(1, []() -> int {
                  llvm_unreachable("Should never be called!");
                }).isInserted());
  // Update does invoke the callback.
  auto IResult = M.update(1, [] { return 101; });
  EXPECT_FALSE(IResult.isInserted());
  EXPECT_EQ(101, IResult.getValue());
  EXPECT_EQ(101, *M[1]);
}

TEST(MapTest, BasicRef) {
  Map<int, int, 4> RealM;
  using ViewT = MapView<int, int>;
  using RefT = MapRef<int, int>;

  RefT M = RealM;

  EXPECT_FALSE(M.contains(42));
  EXPECT_EQ(nullptr, M[42]);
  EXPECT_TRUE(M.insert(1, 100).isInserted());
  ASSERT_TRUE(M.contains(1));
  ASSERT_TRUE(ViewT(M).contains(1));
  auto Result = M.lookup(1);
  EXPECT_TRUE(Result);
  EXPECT_EQ(1, Result.getKey());
  EXPECT_EQ(100, Result.getValue());
  EXPECT_EQ(100, *M[1]);
  Result = ViewT(M).lookup(1);
  EXPECT_TRUE(Result);
  EXPECT_EQ(1, Result.getKey());
  EXPECT_EQ(100, Result.getValue());
  EXPECT_EQ(100, *ViewT(M)[1]);
  // Reinsertion doesn't change the value.
  auto IResult = M.insert(1, 101);
  EXPECT_FALSE(IResult.isInserted());
  EXPECT_EQ(100, IResult.getValue());
  EXPECT_EQ(100, *M[1]);
  // Update does change the value.
  IResult = M.update(1, 101);
  EXPECT_FALSE(IResult.isInserted());
  EXPECT_EQ(101, IResult.getValue());
  EXPECT_EQ(101, *M[1]);

  // Now fill it up.
  for (int i : llvm::seq(2, 100))
    EXPECT_TRUE(M.insert(i, i * 100).isInserted()) << "Key: " << i;
  for (int i : llvm::seq(1, 100)) {
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + (int)(i == 1), *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 3).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 3, *M[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(RealM.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 3, *RealM[i]) << "Key: " << i;
  }
  EXPECT_FALSE(M.contains(420));

  // Clear back to empty.
  M.clear();
  EXPECT_FALSE(M.contains(42));
  EXPECT_EQ(nullptr, M[42]);

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150))
    EXPECT_TRUE(M.insert(i, i * 100).isInserted()) << "Key: " << i;
  for (int i : llvm::seq(50, 150)) {
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100, *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(RealM.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *RealM[i]) << "Key: " << i;
  }
  EXPECT_FALSE(M.contains(42));
  EXPECT_FALSE(M.contains(420));

  EXPECT_FALSE(M.erase(42));
  EXPECT_TRUE(M.contains(73));
  EXPECT_TRUE(M.erase(73));
  EXPECT_FALSE(M.contains(73));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(M.contains(i));
    EXPECT_TRUE(M.erase(i));
    EXPECT_FALSE(M.contains(i));
  }
  for (int i : llvm::seq(50, 150)) {
    if (i == 73 || (i >= 102 && i < 136))
      continue;
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *M[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(RealM.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *RealM[i]) << "Key: " << i;
  }
  EXPECT_TRUE(M.insert(73, 73 * 100 + 3).isInserted());
  EXPECT_EQ(73 * 100 + 3, *M[73]);

  // Reset back to empty and small.
  M.reset();
  EXPECT_FALSE(M.contains(42));
  EXPECT_EQ(nullptr, M[42]);

  // Refill but with both overlapping and different values, now triggering
  // growth too. Also, use update instead of insert.
  for (int i : llvm::seq(75, 175))
    EXPECT_TRUE(M.update(i, i * 100).isInserted()) << "Key: " << i;
  for (int i : llvm::seq(75, 175)) {
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100, *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100, *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 1).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(RealM.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *RealM[i]) << "Key: " << i;
  }
  EXPECT_FALSE(M.contains(42));
  EXPECT_FALSE(M.contains(420));

  EXPECT_FALSE(M.erase(42));
  EXPECT_TRUE(M.contains(93));
  EXPECT_TRUE(M.erase(93));
  EXPECT_FALSE(M.contains(93));
  for (int i : llvm::seq(102, 136)) {
    EXPECT_TRUE(M.contains(i));
    EXPECT_TRUE(M.erase(i));
    EXPECT_FALSE(M.contains(i));
  }
  for (int i : llvm::seq(75, 175)) {
    if (i == 93 || (i >= 102 && i < 136))
      continue;
    ASSERT_TRUE(M.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
    ASSERT_TRUE(ViewT(M).contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *ViewT(M)[i]) << "Key: " << i;
    EXPECT_FALSE(M.insert(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 1, *M[i]) << "Key: " << i;
    EXPECT_FALSE(M.update(i, i * 100 + 2).isInserted()) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *M[i]) << "Key: " << i;

    // Also check that the real map observed all of these changes.
    ASSERT_TRUE(RealM.contains(i)) << "Key: " << i;
    EXPECT_EQ(i * 100 + 2, *RealM[i]) << "Key: " << i;
  }
  EXPECT_TRUE(M.insert(93, 93 * 100 + 3).isInserted());
  EXPECT_EQ(93 * 100 + 3, *M[93]);
}

}  // namespace
}  // namespace Carbon::Testing
