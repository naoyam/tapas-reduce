#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

#include <tapas/test.h>
#include <tapas/mpi_util.h>

SETUP_TEST;

int mpi_rank = -1;
int mpi_size = -1;

int sample_func(int src, int dst) {
  src++; dst++;
  if (src == dst)
    return 0;
  else if (src > dst)
    return src * dst;
  else
    return dst % src;
}

void Test_MPI_Alltoall() {
  // Perform MPI_Alltoall using tapas::mpi_util::Alltoallv().
  // Process src sends a single integer value (= sample_func(src,dst)) to
  // the process dst.

  std::vector<int> send_buf(mpi_size), recv_buf, dest(mpi_size), src(mpi_size);
  for (size_t i = 0; i < mpi_size; i++) {
    send_buf[i] = sample_func(mpi_rank, i);
    dest[i] = i;
  }

  tapas::mpi::Alltoallv(send_buf, dest, recv_buf, src, MPI_COMM_WORLD);

  std::vector<int> recv_buf_should(mpi_size), src_should(mpi_size);

  for (size_t i = 0; i < mpi_size; i++) {
    recv_buf_should[i] = sample_func(i, mpi_rank);
    src_should[i] = i;
  }

  ASSERT_EQ(src_should, src);
  ASSERT_EQ(recv_buf_should, recv_buf);
}

void Test_MPI_Alltoallv() {
  // Perform MPI_Alltoallv.
  // Process src send
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  
  Test_MPI_Alltoall();
  Test_MPI_Alltoallv();

  TEST_REPORT_RESULT();
  
  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
