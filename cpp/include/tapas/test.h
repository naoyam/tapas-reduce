/**
 * \file test.cpp A very simple unittest library for Tapas
 */

#include <iostream>
#include <string>
#include <sstream>
#include <iomanip>

#include <mpi.h>

#define SETUP_TEST                              \
  namespace test_app {                          \
  int succ_cnt = 0;                             \
  int fail_cnt = 0;                             \
  std::vector<std::string> fail_log;            \
  }                                             \

#define ASSERT_EQ(should, actual) do {              \
    auto __s = should;                              \
    auto __a = actual;                              \
    if (__s == __a) {                               \
      test_app::succ_cnt++;                         \
    } else {                                        \
      std::stringstream ss;                         \
      ss << "ASSERT_EQUAL failed in "               \
         << __FILE__ << ":" << __LINE__ << " "      \
         << __PRETTY_FUNCTION__                     \
         << std::endl;                              \
      ss << "\tshould = '" << #should << "' (= "    \
         << tapas::test::ToString(__s) << "), "     \
         << "\tactual = '" << #actual << "' (= "    \
         << tapas::test::ToString(__a) << ").";     \
      ss << std::endl;                              \
      test_app::fail_log.push_back(ss.str());       \
      test_app::fail_cnt++;                         \
    }                                               \
  } while(0)

#define ASSERT_TRUE(exp) do {                         \
    auto __e = exp;                                   \
    if (__e) {                                        \
      test_app::succ_cnt++;                           \
    } else {                                          \
      std::stringstream ss;                           \
      ss << "ASSERT_TRUE failed in "                  \
         << __FILE__ << ":" << __LINE__ << " "        \
         << __PRETTY_FUNCTION__ << "." << std::endl;  \
      ss << "\tactual = '" << #exp << "' (= "         \
         << tapas::test::ToString(exp) << ").";       \
      ss << std::endl;                                \
      test_app::fail_log.push_back(ss.str());         \
      test_app::fail_cnt++;                           \
    }                                                 \
  } while(0)

#define ASSERT_FALSE(exp) do {                        \
    auto e = (exp);                                   \
    if (!e) {                                         \
      test_app::succ_cnt++;                           \
    } else {                                          \
      std::stringstream ss;                           \
      ss << "ASSERT_FALSE failed in "                 \
         << __FILE__ << ":" << __LINE__ << " "        \
         << __PRETTY_FUNCTION__ << "." << std::endl;  \
      ss << "\tactual = '" << #exp << "' (= "         \
         << tapas::test::ToString(e) << "), "         \
         << "which is not 0";                         \
      ss << std::endl;                                \
      test_app::fail_log.push_back(ss.str());         \
      test_app::fail_cnt++;                           \
    }                                                 \
  } while(0)

#define TEST_REPORT_RESULT() do {                                       \
    tapas::test::ReportResult(test_app::succ_cnt, test_app::fail_cnt, test_app::fail_log); \
  } while(0)

#define TEST_SUCCESS() (test_app::fail_cnt == 0)

namespace tapas {
namespace test {

const constexpr char * RESET   = "\033[0m";
const constexpr char * BLACK   = "\033[30m";      /* Black */
const constexpr char * RED     = "\033[31m";      /* Red */
const constexpr char * GREEN   = "\033[32m";      /* Green */
const constexpr char * YELLOW  = "\033[33m";      /* Yellow */
const constexpr char * BLUE    = "\033[34m";      /* Blue */
const constexpr char * MAGENTA = "\033[35m";      /* Magenta */
const constexpr char * CYAN    = "\033[36m";      /* Cyan */
const constexpr char * WHITE   = "\033[37m";      /* White */
const constexpr char * BOLDBLACK   = "\033[1m\033[30m";      /* Bold Black */
const constexpr char * BOLDRED     = "\033[1m\033[31m";      /* Bold Red */
const constexpr char * BOLDGREEN   = "\033[1m\033[32m";      /* Bold Green */
const constexpr char * BOLDYELLOW  = "\033[1m\033[33m";      /* Bold Yellow */
const constexpr char * BOLDBLUE    = "\033[1m\033[34m";      /* Bold Blue */
const constexpr char * BOLDMAGENTA = "\033[1m\033[35m";      /* Bold Magenta */
const constexpr char * BOLDCYAN    = "\033[1m\033[36m";      /* Bold Cyan */
const constexpr char * BOLDWHITE   = "\033[1m\033[37m";      /* Bold White */

const constexpr char * CLEAR = "\033[2J";  // clear screen escape code 

template<class T>
std::string ToString(const T& val) {
  std::stringstream ss;
  ss << val;
  return ss.str();
}

template<class T>
std::string ToString(const std::vector<T>& val) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < val.size(); i++) {
    ss << ToString(val[i]);
    if (i < val.size() - 1) {
      ss << ", ";
    }
  }

  ss << "]";
  return ss.str();
}

void ReportResult(int &succ_cnt, int &fail_cnt, const std::vector<std::string> &fail_log) {
  int size = 0;
  int rank = 0;

  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int succ_all = -1;
  int fail_all = -1;
  
  MPI_Allreduce(&succ_cnt, &succ_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&fail_cnt, &fail_all, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << std::fixed << std::setw(5)
              << (succ_all + fail_all) << " tests ran in total" << std::endl;
    // print number of successed tests.
    std::cout << RESET << GREEN << std::fixed << std::setw(5);
    std::cout << succ_all << " passed." << RESET << std::endl;
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  
  for (int i = 0; i < size; i++) {
    if (i == rank && fail_cnt > 0) {
      std::cout << "In MPI Rank " << i << std::endl;
      
      std::cout << RESET << RED << std::fixed << std::setw(5);
      std::cout << fail_cnt << " failed." << RESET << std::endl;
      for (int i = 0; i < fail_log.size(); i++) {
        std::cout << "  " << "[" << i << "]  " << fail_log[i];
      }
    }
    MPI_Barrier(MPI_COMM_WORLD);
  }

  succ_cnt = succ_all;
  fail_cnt = fail_all;

  return;
}

}
}
