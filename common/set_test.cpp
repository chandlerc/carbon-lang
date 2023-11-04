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
void expectUnorderedElementsAre(SetT&& s, MatcherRangeT element_matchers) {
  // Now collect the elements into a container.
  using KeyT = typename std::remove_reference<SetT>::type::KeyT;
  std::vector<KeyT> entries;
  s.ForEach([&entries](KeyT& k) {
    entries.push_back(k);
  });

  // Use the GoogleMock unordered container matcher to validate and show errors
  // on wrong elements.
  EXPECT_THAT(entries, UnorderedElementsAreArray(element_matchers));
}

// Allow directly using an initializer list.
template <typename SetT, typename MatcherT>
void expectUnorderedElementsAre(
    SetT&& s, std::initializer_list<MatcherT> element_matchers) {
  std::vector<MatcherT> element_matchers_storage = element_matchers;
  expectUnorderedElementsAre(s, element_matchers_storage);
}

template <typename RangeT, typename ...RangeTs>
auto MakeElements(RangeT&& range, RangeTs&& ...ranges) {
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
  Set<int, 4> m;
  using BaseT = SetBase<int>;
  using ViewT = SetView<int>;

  EXPECT_FALSE(m.Contains(42));
  EXPECT_TRUE(m.Insert(1).is_inserted());
  EXPECT_TRUE(m.Contains(1));
  EXPECT_TRUE(ViewT(m).Contains(1));
  auto result = m.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  result = ViewT(m).Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  auto i_result = m.Insert(1);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_TRUE(m.Contains(1));

  // Verify all the elements.
  expectUnorderedElementsAre(m, {1});

  // Fill up the small buffer but don't overflow it.
  for (int i : llvm::seq(2, 5)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 5)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());
  }
  EXPECT_FALSE(m.Contains(5));

  // Verify all the elements.
  expectUnorderedElementsAre(m, {1, 2, 3, 4});
  expectUnorderedElementsAre(*static_cast<BaseT*>(&m), {1, 2, 3, 4});
  expectUnorderedElementsAre(ViewT(m), {1, 2, 3, 4});

  // Erase some entries from the small buffer.
  EXPECT_FALSE(m.Erase(42));
  EXPECT_TRUE(m.Erase(2));
  EXPECT_TRUE(m.Contains(1));
  EXPECT_FALSE(m.Contains(2));
  EXPECT_TRUE(m.Contains(3));
  EXPECT_TRUE(m.Contains(4));
  EXPECT_TRUE(m.Erase(1));
  EXPECT_FALSE(m.Contains(1));
  EXPECT_FALSE(m.Contains(2));
  EXPECT_TRUE(m.Contains(3));
  EXPECT_TRUE(m.Contains(4));
  EXPECT_TRUE(m.Erase(4));
  EXPECT_FALSE(m.Contains(1));
  EXPECT_FALSE(m.Contains(2));
  EXPECT_TRUE(m.Contains(3));
  EXPECT_FALSE(m.Contains(4));
  // Fill them back in.
  EXPECT_TRUE(m.Insert(1).is_inserted());
  EXPECT_TRUE(m.Insert(2).is_inserted());
  EXPECT_TRUE(m.Insert(4).is_inserted());
  EXPECT_TRUE(m.Contains(1));
  EXPECT_TRUE(m.Contains(2));
  EXPECT_TRUE(m.Contains(3));
  EXPECT_TRUE(m.Contains(4));

  // Now fill up the first control group.
  for (int i : llvm::seq(5, 14)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 14)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());
    EXPECT_TRUE(m.Contains(i));
  }
  EXPECT_FALSE(m.Contains(42));

  // Verify all the elements.
  expectUnorderedElementsAre(m, {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13});

  // Now fill up several more control groups.
  for (int i : llvm::seq(14, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());
    EXPECT_TRUE(m.Contains(i));
  }
  EXPECT_FALSE(m.Contains(420));
  expectUnorderedElementsAre(m, MakeElements(llvm::seq(1, 100)));

  // Clear back to empty.
  m.Clear();
  EXPECT_FALSE(m.Contains(42));

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());
    EXPECT_TRUE(m.Contains(i));
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));
  expectUnorderedElementsAre(m, MakeElements(llvm::seq(50, 150)));

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
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());
    EXPECT_TRUE(m.Contains(i));
  }
  EXPECT_TRUE(m.Insert(73).is_inserted());
  EXPECT_TRUE(m.Contains(73));
  expectUnorderedElementsAre(
      m, MakeElements(llvm::seq(50, 102), llvm::seq(136, 150)));

  // Reset back to empty and small.
  m.Reset();
  EXPECT_FALSE(m.Contains(42));

  // Refill but with both overlapping and different values, now triggering
  // growth too.
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());
    EXPECT_TRUE(m.Contains(i));
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));
  expectUnorderedElementsAre(ViewT(m), MakeElements(llvm::seq(75, 175)));

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
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    if (i == 93 || (i >= 102 && i < 136)) {
      continue;
    }
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());
    EXPECT_TRUE(m.Contains(i));
  }
  EXPECT_TRUE(m.Insert(93).is_inserted());
  EXPECT_TRUE(m.Contains(93));
  expectUnorderedElementsAre(
      ViewT(m), MakeElements(llvm::seq(75, 102), llvm::seq(136, 175)));
}

TEST(SetTest, FactoryAPI) {
  Set<int, 4> m;
  EXPECT_TRUE(m.Insert(1, [](int k, void* key_storage) {
                 return new (key_storage) int(k);
               }).is_inserted());
  ASSERT_TRUE(m.Contains(1));
  // Reinsertion doesn't invoke the callback.
  EXPECT_FALSE(m.Insert(1, [](int, void*) -> int* {
                  llvm_unreachable("Should never be called!");
                }).is_inserted());
}

TEST(SetTest, BasicRef) {
  Set<int, 4> real_m;
  using ViewT = SetView<int>;
  using BaseT = SetBase<int>;

  BaseT& m = real_m;

  EXPECT_FALSE(m.Contains(42));
  EXPECT_TRUE(m.Insert(1).is_inserted());
  EXPECT_TRUE(m.Contains(1));
  EXPECT_TRUE(ViewT(m).Contains(1));
  auto result = m.Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  result = ViewT(m).Lookup(1);
  EXPECT_TRUE(result);
  EXPECT_EQ(1, result.key());
  auto i_result = m.Insert(1);
  EXPECT_FALSE(i_result.is_inserted());
  EXPECT_TRUE(m.Contains(1));

  // Now fill it up.
  for (int i : llvm::seq(2, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(1, 100)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());

    // Also check that the real set observed all of these changes.
    EXPECT_TRUE(real_m.Contains(i));
  }
  EXPECT_FALSE(m.Contains(420));

  // Clear back to empty.
  m.Clear();
  EXPECT_FALSE(m.Contains(42));

  // Refill but with both overlapping and different values.
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(50, 150)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());

    // Also check that the real set observed all of these changes.
    EXPECT_TRUE(real_m.Contains(i));
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

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
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());

    // Also check that the real set observed all of these changes.
    EXPECT_TRUE(real_m.Contains(i));
  }
  EXPECT_TRUE(m.Insert(73).is_inserted());
  EXPECT_TRUE(m.Contains(73));

  // Reset back to empty and small with the real map.
  real_m.Reset();
  EXPECT_FALSE(m.Contains(42));

  // Refill but with both overlapping and different values, now triggering
  // growth too.
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Insert(i).is_inserted());
  }
  for (int i : llvm::seq(75, 175)) {
    SCOPED_TRACE(llvm::formatv("Key: {0}", i).str());
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());

    // Also check that the real set observed all of these changes.
    EXPECT_TRUE(real_m.Contains(i));
  }
  EXPECT_FALSE(m.Contains(42));
  EXPECT_FALSE(m.Contains(420));

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
    EXPECT_TRUE(m.Contains(i));
    EXPECT_TRUE(ViewT(m).Contains(i));
    EXPECT_FALSE(m.Insert(i).is_inserted());

    // Also check that the real set observed all of these changes.
    EXPECT_TRUE(real_m.Contains(i));
  }
  EXPECT_TRUE(m.Insert(93).is_inserted());
  EXPECT_TRUE(m.Contains(93));
  expectUnorderedElementsAre(
      ViewT(m), MakeElements(llvm::seq(75, 102), llvm::seq(136, 175)));
  expectUnorderedElementsAre(
      ViewT(real_m), MakeElements(llvm::seq(75, 102), llvm::seq(136, 175)));
}

}  // namespace
}  // namespace Carbon
