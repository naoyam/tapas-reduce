#include <ostream>

#include <gtest/gtest.h>

#include <tapas/common.h>
#include <tapas/map.h>
#include <tapas/hot.h>

template<class T> using V = std::vector<T>;

#define DEF_STREAM_HELPER(TYPE)                                         \
  template <>                                                           \
  inline void GTestStreamToHelper<std::vector<TYPE>>(std::ostream* os, const std::vector<TYPE>& v) { \
    *os << "[";                                                         \
    for (size_t i = 0; i < v.size(); i++) {                             \
      *os << v[i];                                                      \
      *os << (i == v.size() - 1 ? "" : ",");                            \
    }                                                                   \
    *os << "]";                                                         \
  }

DEF_STREAM_HELPER(int)
DEF_STREAM_HELPER(char)

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

TEST(TestDiff, TestDiff) {
  using tapas::ContainerDiff;
  
}
