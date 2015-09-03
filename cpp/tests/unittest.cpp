#include <ostream>

#include <gtest/gtest.h>

#include <tapas/common.h>
#include <tapas/map.h>
#include <utility>
#include <set>

template<class T> using V = std::vector<T>;

#include "test_morton_key.cpp"

TEST(TestAlgorithm, TestSortByKeys) {
  V<char> vals {'2', '3', '0', '4', '5', '1'};
  V<int>  keys { 2,   3,   0,   4,   5,   1};
  V<char> ans_vals  {'0', '1', '2', '3', '4', '5'};
  V<int>  ans_keys  { 0,   1,   2,   3,   4,   5 };

  tapas::SortByKeys(keys, vals);
  ASSERT_EQ(ans_vals, vals);
  ASSERT_EQ(ans_keys, keys);
}

#if 0
// SetUnion is no longer used
TEST(TestMap, TestSetUnion) {
  using tapas::hot::SetUnion;
  typedef std::vector<int> Vi;

  {
    Vi a {1,2,3};
    Vi b {4,5,6};
    Vi ans {1,2,3,4,5,6};
    ASSERT_TRUE(ans == SetUnion(a,b));
  }
  {
    Vi a {};
    Vi b {};
    Vi ans {};
    ASSERT_TRUE(ans == SetUnion(a,b));
  }
  {
    Vi a {1,2,3};
    Vi b {};
    Vi ans {1,2,3};
    ASSERT_TRUE(ans == SetUnion(a,b));
  }
  {
    Vi a {};
    Vi b {4,5,6};
    Vi ans {4,5,6};
    ASSERT_TRUE(ans == SetUnion(a,b));
  }
  {
    Vi a {1,3,5};
    Vi b {2,4,6};
    Vi ans {1,2,3,4,5,6};
    ASSERT_TRUE(ans == SetUnion(a,b));
  }
}
#endif

TEST(TestSet, TestIfSetIsSorted) {
  // Test if std::set (not unordered_set) iterates over the elements in a sorted fasion.
  srand(0);
  const int N = 100;
  std::set<std::pair<int, double>> S;

  for (int i = 0; i < N; i++) {
    S.insert(std::make_pair(rand(), drand48()));
  }

  std::vector<int> v;
  for (auto &&itr : S) {
    v.push_back(itr.first);
  }

  // check if v is sorted.
  for (size_t i = 0; i < v.size()-1; i++) {
    ASSERT_TRUE(v[i] <= v[i+1]);
  }
}

