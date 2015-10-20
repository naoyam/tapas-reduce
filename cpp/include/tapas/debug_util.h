#ifndef TAPAS_DEBUG_UTIL_H_
#define TAPAS_DEBUG_UTIL_H_

#include <string>

#include "tapas/common.h"
#include "tapas/basic_types.h"

#ifdef USE_MPI
# include <mpi.h>
#endif

namespace tapas {
namespace debug {

template <int DIM, class FP, class BT>
void PrintBodies(const typename BT::type *b, int nb, std::ostream &os) {
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < DIM; ++j) {    
      FP p = *ParticlePosOffset<DIM, FP, BT::pos_offset>::pos((void *)&(b[i]), j);
      os << p << " ";
    }
    os << std::endl;
  }
}

template <class T>
std::string ToStr(T v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

#ifdef USE_MPI

// Debug helper function
template<class F>
void BarrierExec(F func) {
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < size; i++) {
    if (rank == i) {
      func(rank, size);
    }
    usleep(1000000);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

#else

template<class F>
void BarrierExec(F func) {
  func(0, 1);
}

#endif

} // debug
} // tapas

#endif // TAPAS_DEBUG_UTIL_H_
