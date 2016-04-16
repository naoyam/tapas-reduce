
#include <utility>
#include <set>

#ifndef TAPAS_DEBUG
#define TAPAS_DEBUG 1 // always use TAPAS_DEBUG
#endif

#include <tapas/common.h>
#include <tapas/test.h>
#include <tapas/geometry.h>

SETUP_TEST;

using V1 = tapas::Vec<1, double>;
using V2 = tapas::Vec<2, double>;
using Reg1 = tapas::Region<1, double>;
using Reg2 = tapas::Region<2, double>;

bool Close(double a, double b, const char *, int) {
  return fabs(a - b) < 1e-6;
}

bool Close(double a, V1 b, const char *, int) {
  return fabs(a - b[0]) < 1e-6;
}

bool Close(V1 a, double b, const char *, int) {
  return fabs(a[0] - b) < 1e-6;
}

bool Close(V1 a, V1 b, const char *file, int line) {
  bool ret = fabs(a[0] - b[0]) < 1e-10;
  if (!ret) {
    std::cerr << file << ":" << line << " Close(): Not close: a = " << a << ", b = " << b << std::endl;
  }
  return ret;
}

void Test_Center2d() {
  {
    V2 tmax = {0,0}, tmin = {-1,-1};
    V2 smax = {1,1}, smin = { 0, 0};
    Reg2 t(tmin, tmax), s(smin, smax);

    // Case 1
    double dist = tapas::Distance<2, tapas::CenterClass, double>::CalcApprox(t, s);
    double dist_ans = 2;
    ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));
  }

  {
    // Case 2
    V2 tmax = {0,0}, tmin = {-1,-1};
    V2 smax = {1,1}, smin = {0.5, 0.5};
    Reg2 t(tmin, tmax), s(smin, smax);

    double dist = tapas::Distance<2, tapas::CenterClass, double>::CalcApprox(t, s);
    double dist_ans = 2;
    ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));
  }

  {
    // Case 3
    V2 tmax = {0,0}, tmin = {-1,-1};
    V2 smax = {1,0}, smin = {0.5, - 0.5};
    Reg2 t(tmin, tmax), s(smin, smax);

    double dist = tapas::Distance<2, tapas::CenterClass, double>::CalcApprox(t, s);
    double dist_ans = 1;
    ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));
  }

}

void Test_Center1d() {
  V1 tmax = {1}, tmin = {-1};
  V1 dist, dist_ans;
  V1 sctr, tctr;
  V1 R = 0.5;
  Reg1 t(tmin, tmax), s;
  using D = tapas::Distance<1, tapas::CenterClass, double>;

  // Case 1
  // |---+---|                 src
  //   |---+-----------------| trg
  //  -1                     1
  sctr = -0.8;
  s.min() = sctr - R/2;
  s.max() = sctr + R/2;
  tctr = -0.75; // answer : -1 + 0.25
  dist = D::CalcApprox(t, s);
  dist_ans = (tctr - sctr) * (tctr - sctr);
  ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));

  // Case 2
  //     |---+---|             src
  //   |-----+---------------| trg
  //  -1                     1
  sctr = -0.7;
  s.min() = sctr - R/2;
  s.max() = sctr + R/2;
  tctr = sctr; // answer
  dist = D::CalcApprox(t, s);
  dist_ans = 0 * 0;
  ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));

  // Case 3
  //               |---+---|   src
  //   |---------------+-----| trg
  //  -1                     1
  sctr = 0.4;
  s.min() = sctr - R/2;
  s.max() = sctr + R/2;
  tctr = sctr; // answer
  dist = D::CalcApprox(t, s);
  dist_ans = 0 * 0;
  ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));

  // Case 4
  //                   |---+---| src
  //   |-----------------+---|   trg
  //  -1                     1
  sctr = 0.4;
  s.min() = sctr - R/2;
  s.max() = sctr + R/2;
  tctr = sctr; // answer
  dist = D::CalcApprox(t, s);
  dist_ans = 0 * 0;
  ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));

  // Case 5
  //                           |---+---| src
  //   |-----------------+---|           trg
  //  -1                     1
  sctr = 0.4;
  s.min() = sctr - R/2;
  s.max() = sctr + R/2;
  tctr = sctr; // answer
  dist = D::CalcApprox(t, s);
  dist_ans = 0 * 0;
  ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));

  // Case 6
  // |-------------+-------------| src
  //   |----------+----------|   trg
  //  -1                     1
  R = 2.5;
  sctr = 0.1;
  s.min() = sctr - R/2;
  s.max() = sctr + R/2;
  tctr = 0.0;
  dist = D::CalcApprox(t, s);
  dist_ans = 0;
  //std::cout << "dist = " << sqrt(dist[0]) << std::endl;
  //std::cout << "ditt_ans = " << sqrt(dist_ans[0]) << std::endl;
  ASSERT_TRUE(Close(dist, dist_ans, __FILE__, __LINE__));
}

void Test_Separated() {
  {
    V1 xmax = {1.0}, xmin = {0.0}, ymax = {3.0}, ymin = {1.1};
    Reg1 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(tapas::Separated(x, y));
    ASSERT_TRUE(tapas::Separated(y, x));
  }

  {
    V1 xmax = {1.0}, xmin = {0.0}, ymax = {3.0}, ymin = {1.1};
    Reg1 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(tapas::Separated(x, y));
    ASSERT_TRUE(tapas::Separated(y, x));
  }

  {
    V1 xmax = {1.0}, xmin = {0.0}, ymax = {3.0}, ymin = {1.1};
    Reg1 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(!tapas::Separated(x, x));
    ASSERT_TRUE(!tapas::Separated(y, y));
  }

  {
    V2 xmax = { 1, 1};
    V2 xmin = { 0, 0};
    V2 ymax = { 0, 0};
    V2 ymin = {-1,-1};
    Reg2 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(tapas::Separated(x, y));
    ASSERT_TRUE(tapas::Separated(y, x));
  }

  {
    V2 xmax = { 2, 0};
    V2 xmin = { 1,-1};
    V2 ymax = { 0, 0};
    V2 ymin = {-1,-1};
    Reg2 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(tapas::Separated(x, y));
    ASSERT_TRUE(tapas::Separated(y, x));
  }

  {
    V2 xmax = { 1, 1};
    V2 xmin = { -0.1, -0.1};
    V2 ymax = { 0, 0};
    V2 ymin = {-1,-1};
    Reg2 x(xmin, xmax), y(ymin, ymax);
    ASSERT_TRUE(!tapas::Separated(x, y));
    ASSERT_TRUE(!tapas::Separated(y, x));
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Test_Center1d();
  Test_Center2d();
  Test_Separated();

  TEST_REPORT_RESULT();

  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
