#include <ostream>

#include <gtest/gtest.h>

#include <tapas/map.h>
#include <tapas/hot.h>

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

  tapas::morton_hot::SortByPermutations(perms, vals);
  ASSERT_EQ(ans, vals);
}

template<class T>
void CheckMatching(const T& S, const T& R) {
  // Check that all send/recv pairs match.
  // From senders' view
  for (auto &s : S) {
    // Check all the recievers of s receive from s.
    auto receivers_of_s = SendRecvMapping(S, R, s);
    
    for (auto &r: receivers_of_s) {
      auto senders_of_r = SendRecvMapping(S, R, r);
      ASSERT_EQ(1, senders_of_r.size());
      ASSERT_EQ(s, senders_of_r[0]);
    }
  }

  // From receiver's view
  for (auto &&r : R) {
    std::stringstream ss;
    ss << "Receiver's view: r=" << r;
    SCOPED_TRACE(ss.str());
    
    // Check r's sender sends to r.
    auto senders = SendRecvMapping(S, R, r);
    ASSERT_EQ(1, senders.size());
    auto s = senders[0];
    auto receivers_of_s = SendRecvMapping(S, R, s);

    auto beg = std::begin(receivers_of_s);
    auto end = std::end(receivers_of_s);
    auto i = std::find(beg, end, r);
    ASSERT_TRUE(beg <= i && i < end);
  }
}


TEST(TestMap, TestSendRecvMapping1) {
  typedef std::vector<int> vec;

  // We need to check 3 cases:
  //   r = |R|
  //   s = |S|

  // Case 1 : s > r
  {
    // Case 1-1:  s % r == 0
    vec S {0, 1, 2, 3};
    vec R {4, 5};
    SCOPED_TRACE("S={0,1,2,3} R={4,5}");
    CheckMatching(S, R);
  }

  return;
  {
    // Case 1-2:  s % r != 0
    vec S {0, 1, 2, 3, 9};
    vec R {4, 5};
    SCOPED_TRACE("S={0,1,2,3,9} R={4,5}");
    CheckMatching(S, R);
  }

  // Case 2 : s == r
  {
    SCOPED_TRACE("S={0,1,2} R={3,4,5}");
    vec S {0, 1, 2};
    vec R {3, 4, 5};
    CheckMatching(S, R);
  }
  
  // Case 3 : s < r
  {
    // Case 3-1:  r % s == 0
    SCOPED_TRACE("S={0,1} R={2,3,4,5}");
    vec S {0, 1};
    vec R {2, 3, 4, 5};
    CheckMatching(S, R);
  }

  {
    // Case 3-2: r % s != 0
    SCOPED_TRACE("S={0,1} R={2,3,4,5,6}");
    vec S {0, 1};
    vec R {2, 3, 4, 5, 6};
    CheckMatching(S, R);
  }
}

TEST(TestMap, TestSendRecvMapping2) {
  // regression test
  typedef std::vector<int> vec;

  vec S {0,1,2};
  vec R {3,4,5,6,7};
  SCOPED_TRACE("TestSendRecvMapping2 S={0,1,2}, R={3,4,5,6,7}");
  CheckMatching(S, R);
}

TEST(TestMap, TestSendRecvMapping3) {
  // regression test 2
  typedef std::vector<int> vec;

  vec S = {4,5};     // senders
  vec R = {0,1,2,3}; // receivers
  SCOPED_TRACE("TestSendRecvMapping3 S={4,5}, R={0,1,2,3}");
  CheckMatching(S, R);
}

TEST(TestMap, TestSendRecvMapping4) {
  // regression test 3
  typedef std::vector<int> vec;

  vec S = {2,3,4};     // senders
  vec R = {5,6,7,8}; // receivers

  SCOPED_TRACE("TestSendRecvMapping4 S={2,3,4}, R={5,6,7,8}");
  CheckMatching(S, R);
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
    std::vector<int> a   {};
    std::vector<int> b   {1,3};
    std::vector<int> ans {};
    ASSERT_TRUE(ans == SetDiff(a,b));
  }

  {
    std::vector<int> a   {1,2,3};
    std::vector<int> b   {1,3,4,5};
    std::vector<int> ans {2};
    ASSERT_TRUE(ans == SetDiff(a,b));
  }
  
  {
    std::vector<int> a   {1,2,3};
    std::vector<int> b   {};
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

