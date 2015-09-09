#include <iostream>
#include <string>
#include <vector>
#include <iterator>
#include <algorithm>
#include <functional>

#include <mpi.h>

#include <tapas/mpi_util.h>
#include <tapas/test.h>

SETUP_TEST;

void Test_InclusiveScan() {
  {
    std::vector<int> in = {1,2,3,4,5};
    std::vector<int> out;
    std::vector<int> should_be = {1,3,6,10,15};

    tapas::util::inclusive_scan(in.begin(), in.end(),
                                back_inserter(out), std::plus<int>());
    ASSERT_EQ(should_be, out);
  }

  {
    std::vector<int> in = {1,0,1,0,1,0};
    std::vector<int> out;
    std::vector<int> should_be = {1,1,2,2,3,3};

    tapas::util::inclusive_scan(in.begin(), in.end(),
                                back_inserter(out), std::plus<int>());
    ASSERT_EQ(should_be, out);
  }
  
  {
    std::vector<int> in = {1,2,3,4,5};
    std::vector<int> out(in.size());
    std::vector<int> should_be = {1,3,6,10,15};
    
    tapas::util::inclusive_scan(in.begin(), in.end(), out.begin(), std::plus<int>());
    ASSERT_EQ(should_be, out);
  }
  
  {
    std::vector<int> in = {1,2,3,4,5};
    std::vector<int> out(in.size());
    std::vector<int> should_be = {1,-1,-4,-8,-13};
    
    tapas::util::inclusive_scan(in.begin(), in.end(), out.begin(), std::minus<int>());
    ASSERT_EQ(should_be, out);
  }
}

void Test_ExclusiveScan() {
  {
    std::vector<int> in = {1,2,3,4,5};
    std::vector<int> out;
    std::vector<int> should_be = {0, 1,3,6,10};

    tapas::util::exclusive_scan(in.begin(), in.end(),
                                back_inserter(out), std::plus<int>());
    ASSERT_EQ(should_be.size(), out.size());
    ASSERT_EQ(should_be, out);
  }

  {
    std::vector<int> in = {1,0,1,0,1,0};
    std::vector<int> out;
    std::vector<int> should_be = {0, 1,1,2,2,3};

    tapas::util::exclusive_scan(in.begin(), in.end(),
                                back_inserter(out), std::plus<int>());
    ASSERT_EQ(should_be.size(), out.size());
    ASSERT_EQ(should_be, out);
  }

  {
    std::vector<int> in = {1,2,3,4,5};
    std::vector<int> out(in.size());
    std::vector<int> should_be = {0,1,3,6,10};
    
    tapas::util::exclusive_scan(in.begin(), in.end(), out.begin(), std::plus<int>());
    ASSERT_EQ(should_be.size(), out.size());
    ASSERT_EQ(should_be, out);
  }

  {
    std::vector<int> in = {1,2,3,4,5};
    std::vector<int> out(in.size());
    std::vector<int> should_be = {0,-1,-3,-6,-10};
    
    tapas::util::exclusive_scan(in.begin(), in.end(), out.begin(), std::minus<int>());
    ASSERT_EQ(should_be.size(), out.size());
    ASSERT_EQ(should_be, out);
  }
}

// test 2 : Send custom struct.
struct T {
  int rank;
  int rank2;
  bool operator==(const T& rhs) const {
    return rank == rhs.rank && rank2 == rhs.rank2;
  }
};

std::ostream& operator<<(std::ostream &os, const T& t) {
  os << "{" << t.rank << "," << t.rank2 << "}";
  return os;
}

void Test_Allgatherv() {
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  // Test MPI_Allgatherv.
  {
    // test 1 : rank i send [i,i,..,i] (length = i)
    std::vector<int> sendbuf(rank);
    std::vector<int> recvbuf;
    std::vector<int> should_be;
    for (size_t i = 0; i < sendbuf.size(); i++) { sendbuf[i] = rank; }
    for (int r = 0; r < size; r++) {
      for (int i = 0; i < r; i++) {
        should_be.push_back(r);
      }
    }
    
    tapas::mpi::Allgatherv(sendbuf, recvbuf, MPI_COMM_WORLD);
    int total = (size-1) * size / 2;
    ASSERT_EQ(total, recvbuf.size());
    ASSERT_EQ(should_be, recvbuf);
  }

  {
    std::vector<T> sendbuf(rank);
    std::vector<T> recvbuf;
    std::vector<T> should_be;
    for (size_t i = 0; i < sendbuf.size(); i++) {
      T t = {rank, rank*2};
      sendbuf[i] = t;
    }
    for (int r = 0; r < size; r++) {
      for (int i = 0; i < r; i++) {
        T t = {r, r * 2};
        should_be.push_back(t);
      }
    }
    
    tapas::mpi::Allgatherv(sendbuf, recvbuf, MPI_COMM_WORLD);
    int total = (size-1) * size / 2;
    ASSERT_EQ(total, recvbuf.size());
    ASSERT_EQ(should_be, recvbuf);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  
  Test_InclusiveScan();
  Test_ExclusiveScan();
  Test_Allgatherv();

  TEST_REPORT_RESULT();
  
  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
