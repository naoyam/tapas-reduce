#include <ostream>

#include <gtest/gtest.h>

#include <tapas/map.h>
#include <tapas/morton_hot.h>

template<class T> using V = std::vector<T>;

using tapas::morton_hot::SendRecvMapping;

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


TEST(TestSort, TestSortByPermutations) {
  V<char> vals  {'2', '3', '0', '4', '5', '1'};
  V<char> ans   {'0', '1', '2', '3', '4', '5'};
  V<int>  perms { 2,   3,   0,   4,   5,   1};

  tapas::morton_common::SortByPermutations(perms, vals);
  ASSERT_EQ(ans, vals);
}

TEST(TestMap, TestSendRecvMapping1) {
  typedef std::vector<int> vec;

  // We need to check 3 cases:
  //   r = |R|
  //   s = |S|

  // Case 1 : s > r
  {
    {
      // Case 1-1:  s % r == 0
      vec S {0, 1, 2, 3};
      vec R {4, 5};
      
      ASSERT_EQ(vec({4}),    SendRecvMapping(S, R, 0));
      ASSERT_EQ(vec({5}),    SendRecvMapping(S, R, 1));
      ASSERT_EQ(vec({}),     SendRecvMapping(S, R, 2));
      ASSERT_EQ(vec({}),     SendRecvMapping(S, R, 3));
      ASSERT_EQ(vec({0}),    SendRecvMapping(S, R, 4));
      ASSERT_EQ(vec({1}),    SendRecvMapping(S, R, 5));
    }
    {
      // Case 1-2:  s % r != 0
      vec S {0, 1, 2, 3, 9};
      vec R {4, 5};
      
      ASSERT_EQ(vec({4}),    SendRecvMapping(S, R, 0));
      ASSERT_EQ(vec({5}),    SendRecvMapping(S, R, 1));
      ASSERT_EQ(vec({}),     SendRecvMapping(S, R, 2));
      ASSERT_EQ(vec({}),     SendRecvMapping(S, R, 3));
      ASSERT_EQ(vec({}),     SendRecvMapping(S, R, 9));
      ASSERT_EQ(vec({0}),    SendRecvMapping(S, R, 4));
      ASSERT_EQ(vec({1}),    SendRecvMapping(S, R, 5));
    }
  }

  // Case 2 : s == r
  {
    vec S {0, 1, 2};
    vec R {3, 4, 5};

    ASSERT_EQ(vec({3}),    SendRecvMapping(S, R, 0));
    ASSERT_EQ(vec({4}),    SendRecvMapping(S, R, 1));
    ASSERT_EQ(vec({5}),    SendRecvMapping(S, R, 2));
    ASSERT_EQ(vec({0}),    SendRecvMapping(S, R, 3));
    ASSERT_EQ(vec({1}),    SendRecvMapping(S, R, 4));
    ASSERT_EQ(vec({2}),    SendRecvMapping(S, R, 5));
  }
  
  // Case 3 : s < r
  {
    {
      // Case 3-1:  r % s == 0
      vec S {0, 1};
      vec R {2, 3, 4, 5};

      ASSERT_EQ(vec({2,3}),    SendRecvMapping(S, R, 0));
      ASSERT_EQ(vec({4,5}),    SendRecvMapping(S, R, 1));
      ASSERT_EQ(vec({0}),      SendRecvMapping(S, R, 2));
      ASSERT_EQ(vec({0}),      SendRecvMapping(S, R, 3));
      ASSERT_EQ(vec({1}),      SendRecvMapping(S, R, 4));
      ASSERT_EQ(vec({1}),      SendRecvMapping(S, R, 5));
    }

    {
      // Case 3-2: r % s != 0
      vec S {0, 1};
      vec R {2, 3, 4, 5, 6};

      ASSERT_EQ(vec({2,3,4}),  SendRecvMapping(S, R, 0));
      ASSERT_EQ(vec({5,6}),    SendRecvMapping(S, R, 1));
      ASSERT_EQ(vec({0}),      SendRecvMapping(S, R, 2));
      ASSERT_EQ(vec({0}),      SendRecvMapping(S, R, 3));
      ASSERT_EQ(vec({0}),      SendRecvMapping(S, R, 4));
      ASSERT_EQ(vec({1}),      SendRecvMapping(S, R, 5));
      ASSERT_EQ(vec({1}),      SendRecvMapping(S, R, 6));
    }
  }

  
  vec a {1,2};
  vec b {3,4};

  vec ans1 {1};
  vec ans2 {2};
  vec ans3 {3};
  vec ans4 {4};

  vec res1 = SendRecvMapping(a, b, 1);
  vec res2 = SendRecvMapping(a, b, 2);
  vec res3 = SendRecvMapping(a, b, 3);
  vec res4 = SendRecvMapping(a, b, 4);
    
  ASSERT_TRUE(ans3 == res1);
  ASSERT_TRUE(ans4 == res2);
  ASSERT_TRUE(ans1 == res3);
  ASSERT_TRUE(ans2 == res4);
}

TEST(TestMap, TestSendRecvMapping2) {
  // uni-directional mapping of sender/receivers
  typedef std::vector<int> vec;
  vec X {1,2};
  vec Y {3,4,5,6,7};

  vec ans1 {3,4,5};
  vec ans2 {6,7};

  // X is senders, Y is receivers
  ASSERT_TRUE(ans1 == SendRecvMapping(X, Y, 1));
  ASSERT_TRUE(ans2 == SendRecvMapping(X, Y, 2));

  // Y is senders, X is receivers
  vec ans3 {3};
  vec ans4 {4};
  
  ASSERT_EQ(ans3, SendRecvMapping(Y, X, 1));
  ASSERT_TRUE(ans4 == SendRecvMapping(Y, X, 2));
}

TEST(TestMap, TestSendRecvMapping3) {
  // regression test
  typedef std::vector<int> vec;

  vec X {0};
  vec Y {1, 2};

  vec ans1 {1, 2};
  vec ans2 {0};
  
  vec res1 = SendRecvMapping(X, Y, 0);
  ASSERT_TRUE(ans1 == res1);

  vec res2 = SendRecvMapping(X, Y, 2); // I'm rank 2. Need to receive from 0.
  ASSERT_TRUE(ans2 == res2);

  vec S {0,1,2};
  vec R {3,4,5,6,7};
  vec ans3 {0};
  vec ans4 {0};
  vec ans5 {1};
  vec ans6 {1};
  vec ans7 {2};

  ASSERT_TRUE(ans3 == SendRecvMapping(S, R, 3));
  ASSERT_TRUE(ans4 == SendRecvMapping(S, R, 4));
  ASSERT_TRUE(ans5 == SendRecvMapping(S, R, 5));
  ASSERT_TRUE(ans6 == SendRecvMapping(S, R, 6));
  ASSERT_TRUE(ans7 == SendRecvMapping(S, R, 7));
}

TEST(TestMap, TestSendRecvMapping4) {
  // regression test
  typedef std::vector<int> vec;

  vec S = {4,5};     // senders
  vec R = {0,1,2,3}; // receivers

  vec ans0 = {4};   // rank 0 should recieve from 4
  vec ans1 = {4};   // rank 1 should recieve from 4
  vec ans2 = {5};   // rank 2 should recieve from 5
  vec ans3 = {5};   // rank 3 should recieve from 5
  vec ans4 = {0,1}; // rank 4 should send to 0,1
  vec ans5 = {2,3}; // rank 5 should send to 2,3

  ASSERT_EQ(ans0, SendRecvMapping(S, R, 0));
  ASSERT_EQ(ans1, SendRecvMapping(S, R, 1));
  ASSERT_EQ(ans2, SendRecvMapping(S, R, 2));
  ASSERT_EQ(ans3, SendRecvMapping(S, R, 3));
  ASSERT_EQ(ans4, SendRecvMapping(S, R, 4));
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

