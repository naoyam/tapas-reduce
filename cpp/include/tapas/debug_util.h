#ifndef TAPAS_DEBUG_UTIL_H_
#define TAPAS_DEBUG_UTIL_H_

#include <string>
#include <typeinfo>
#include <cxxabi.h>

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

/**
 * \class DebugStream
 * Debug file stream
 */
class DebugStream {
  std::ostream *fs_;

 public:
  DebugStream(const char *label) : fs_(nullptr) {
#ifdef TAPAS_DEBUG
#if defined(USE_MPI)
    pid_t tid = syscall(SYS_gettid);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#else /* USE_MPI */
    const char *rank="s";
    int tid=0;
#endif /* USE_MPI */
    std::stringstream ss;
    ss << label << "."
       << rank  << "."
       << tid
       << ".stderr.txt";
    fs_ = new std::ofstream(ss.str().c_str(), std::ios_base::app);
#else /* TAPAS_DEBUG */
    (void)label;
    fs_ = new std::stringstream();
#endif /* TAPAS_DEBUG */
  }
  
  ~DebugStream() noexcept {
    assert(fs_ != nullptr);
    fs_->flush();
    delete fs_;
    fs_ = nullptr;
  }

  std::ostream &out() {
    assert(fs_ != nullptr);
    return *fs_;
  }
};

template<class T>
std::string GetClassName() {
  std::string clsname = "";
  
  const std::type_info &ti = typeid(T);
  int stat;
  char *name = abi::__cxa_demangle(ti.name(), 0, 0, &stat);

  if (name != nullptr && stat == 0) {
    clsname = name;
  } else {
    clsname = "<Can't get class name>";
  }

  if (name != nullptr) {
    free(name);
  }

  return clsname;
}


} // debug
} // tapas

#endif // TAPAS_DEBUG_UTIL_H_
