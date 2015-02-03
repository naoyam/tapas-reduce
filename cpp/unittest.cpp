#include <gtest/gtest.h>

#include <tapas/map.h>
#include <tapas/morton_hot.h>

using tapas::GetCellExchangePeer;

int add(int x, int y) {
    return x + y;
}

TEST(TestMap, TestGetCellExchangePeer1) {
  typedef std::vector<int> vec;
  vec a {1,2};
  vec b {3,4};

  vec ans1 {3};
  vec ans2 {4};

  vec res1 = GetCellExchangePeer(a, b, 1);
  vec res2 = GetCellExchangePeer(a, b, 2);
    
  ASSERT_TRUE(ans1 == res1);
  ASSERT_TRUE(ans2 == res2);
}

TEST(TestMap, TestGetCellExchangePeer2) {
  typedef std::vector<int> vec;
  vec a {1,2};
  vec b {3,4,5,6,7};

  vec ans1 {3,4,5};
  vec ans2 {6,7};

  vec res1 = GetCellExchangePeer(a, b, 1);
  vec res2 = GetCellExchangePeer(a, b, 2);
    
  ASSERT_TRUE(ans1 == res1);
  ASSERT_TRUE(ans2 == res2);
}

TEST(TestMap, TestGetCellExchangePeer3) {
  typedef std::vector<int> vec;
  vec a {1,2,3,4,5};
  vec b {6,    7};

  vec ans1 {6};
  vec ans2 {6};
  vec ans3 {6};
  vec ans4 {7};
  vec ans5 {7};

  vec res1 = GetCellExchangePeer(a, b, 1);
  vec res2 = GetCellExchangePeer(a, b, 2);
  vec res3 = GetCellExchangePeer(a, b, 3);
  vec res4 = GetCellExchangePeer(a, b, 4);
  vec res5 = GetCellExchangePeer(a, b, 5);
    
  ASSERT_TRUE(ans1 == res1);
  ASSERT_TRUE(ans2 == res2);
  ASSERT_TRUE(ans3 == res3);
  ASSERT_TRUE(ans4 == res4);
  ASSERT_TRUE(ans5 == res5);
}

TEST(TestMap, TestSetDiff) {
  using tapas::morton_hot::SetDiff;

  {
    std::vector<int> a {1,2,3};
    std::vector<int> b {1,3};
    std::vector<int> ans {2};
    ASSERT_TRUE(ans == SetDiff(a,b));
  }

  {
    std::vector<int> a {};
    std::vector<int> b {1,3};
    std::vector<int> ans {};
    ASSERT_TRUE(ans == SetDiff(a,b));
  }

  {
    std::vector<int> a {1,2,3};
    std::vector<int> b {1,3,4,5};
    std::vector<int> ans {2};
    ASSERT_TRUE(ans == SetDiff(a,b));
  }
  
  {
    std::vector<int> a {1,2,3};
    std::vector<int> b {};
    std::vector<int> ans {1,2,3};
    ASSERT_TRUE(ans == SetDiff(a,b));
  }
}

TEST(TestMap, TestSetUnion) {
  using tapas::morton_hot::SetUnion;
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


