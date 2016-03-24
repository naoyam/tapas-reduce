
#include <utility>
#include <set>

#include <tapas/common.h>
#include <tapas/test.h>
#include <tapas/distance.h>

SETUP_TEST;

using V = tapas::Vec<1, double>;

bool Close(double a, double b) {
  return fabs(a - b) < 1e-6;
}

bool Close(double a, V b) {
  return fabs(a - b[0]) < 1e-6;
}

bool Close(V a, double b) {
  return fabs(a[0] - b) < 1e-6;
}

bool Close(V a, V b) {
  return fabs(a[0] - b[0]) < 1e-10;
}

void Test_Center() {
  V tmax = {1}, tmin = {-1};
  V smax, smin, dist, dist_ans;
  V sctr, tctr;
  const V R = 0.5;

  // Case 1
  // |---+---|                 src
  //   |---+-----------------| trg
  //  -1                     1
  sctr = -0.8;
  smin = sctr - R/2;
  smax = sctr + R/2;
  tctr = -0.75; // answer
  dist = tapas::Distance<tapas::CenterClass, double>::CalcApprox(tmax, tmin, smax, smin);
  dist_ans = 0.05 * 0.05;
  ASSERT_TRUE(Close(dist, dist_ans));
  
  // Case 2
  //     |---+---|             src
  //   |-----+---------------| trg
  //  -1                     1
  sctr = -0.7;
  smin = sctr - R/2;
  smax = sctr + R/2;
  tctr = sctr; // answer
  dist = tapas::Distance<tapas::CenterClass, double>::CalcApprox(tmax, tmin, smax, smin);
  dist_ans = 0 * 0;
  ASSERT_TRUE(Close(dist, dist_ans));

  // Case 3
  //               |---+---|   src
  //   |---------------+-----| trg
  //  -1                     1
  sctr = 0.4;
  smin = sctr - R/2;
  smax = sctr + R/2;
  tctr = sctr; // answer
  dist = tapas::Distance<tapas::CenterClass, double>::CalcApprox(tmax, tmin, smax, smin);
  dist_ans = 0 * 0;
  std::cout << "dist = " << sqrt(dist[0]) << std::endl;
  std::cout << "ditt_ans = " << sqrt(dist_ans[0]) << std::endl;
  ASSERT_TRUE(Close(dist, dist_ans));

  // Case 4
  //                   |---+---| src
  //   |-----------------+---|   trg
  //  -1                     1
  sctr = 0.4;
  smin = sctr - R/2;
  smax = sctr + R/2;
  tctr = sctr; // answer
  dist = tapas::Distance<tapas::CenterClass, double>::CalcApprox(tmax, tmin, smax, smin);
  dist_ans = 0 * 0;
  ASSERT_TRUE(Close(dist, dist_ans));
  
  // Case 5
  //                           |---+---| src
  //   |-----------------+---|           trg
  //  -1                     1
  sctr = 0.4;
  smin = sctr - R/2;
  smax = sctr + R/2;
  tctr = sctr; // answer
  dist = tapas::Distance<tapas::CenterClass, double>::CalcApprox(tmax, tmin, smax, smin);
  dist_ans = 0 * 0;
  ASSERT_TRUE(Close(dist, dist_ans));
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  Test_Center();

  TEST_REPORT_RESULT();
  
  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
