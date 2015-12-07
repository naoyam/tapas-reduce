#ifndef TAPAS_HOT_REPORT_H
#define TAPAS_HOT_REPORT_H

#include <sys/unistd.h> // gethostname

#include <ostream>
#include <iostream>
#include <iomanip>

#include "tapas/hot.h"

namespace tapas {
namespace hot {

namespace {
template<class T> using V = std::vector<T>;
}

struct HostName {
  static const constexpr int kLen = 200;
  char value[kLen];
};

template<class Data>
void Report(const Data &data, std::ostream &os = std::cout) {
  // auto comm = data.mpi_comm_;
  auto comm = MPI_COMM_WORLD;

  const int W = 10; // column width
  const int WS = 14; // column width (scientific notation)
    
  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  HostName hostname;
  gethostname(hostname.value, HostName::kLen);

  // print hostnames
  {
    V<HostName> buf;
    tapas::mpi::Gather(hostname, buf, 0, comm);

    if (rank == 0) {
      os << "---- begin hostnames" << std::endl;
      os << "rank hostname" << std::endl;
      for (size_t i = 0; i < size; i++) {
        os << i << "\t" << buf[i].value << std::endl;
      }
      os << "--- end hostnames" << std::endl;
      os << std::endl;
    }
  }

  // sampling rate
  if (rank == 0) {
    os << std::setprecision(5);
    os << "Sampling Rate " << std::scientific << data.sampling_rate << std::endl;
    os << "NB Total " << data.nb_total << std::endl;
  }

  // number of cells and bodies
  V<index_t> nb_before, nb_after, nleaves, ncells;
  tapas::mpi::Gather(data.nb_before, nb_before, 0, comm);
  tapas::mpi::Gather(data.nb_after,  nb_after,  0, comm);
  tapas::mpi::Gather(data.nleaves,   nleaves,   0, comm);
  tapas::mpi::Gather(data.ncells,    ncells,    0, comm);

  if (rank == 0) {
    os << std::endl;
    os << "---- begin load balancing" << std::endl;
    os << std::setw(5) << std::right << "rank"
       << std::setw(W) << std::right << "nb_before"
       << std::setw(W) << std::right << "nb_after"
       << std::setw(W) << std::right << "nleaves"
       << std::setw(W) << std::right << "ncells"
       << std::endl;
    for (int i = 0; i < size; i++) {
      os << std::setw(5) << std::right << i
         << std::setw(W) << std::right << nb_before[i]
         << std::setw(W) << std::right << nb_after[i]
         << std::setw(W) << std::right << nleaves[i]
         << std::setw(W) << std::right << ncells[i]
         << std::endl;
    }
    os << "---- end load balancing" << std::endl;
    os << std::endl;
  }

  V<double> tree_all, tree_sample, tree_exchange, tree_growlocal, tree_growglobal;
  tapas::mpi::Gather(data.time_tree_all, tree_all, 0, comm);
  tapas::mpi::Gather(data.time_tree_sample, tree_sample, 0, comm);
  tapas::mpi::Gather(data.time_tree_exchange, tree_exchange, 0, comm);
  tapas::mpi::Gather(data.time_tree_growlocal, tree_growlocal, 0, comm);
  tapas::mpi::Gather(data.time_tree_growglobal, tree_growglobal, 0, comm);

  if (rank == 0) {
    os << std::endl;
    os << std::setprecision(5);
    os << "---- begin tree construction" << std::endl;
    os << std::setw(5) << std::scientific << std::right << "rank"
       << std::setw(WS) << std::scientific << std::right << "all"
       << std::setw(WS) << std::scientific << std::right << "sample"
       << std::setw(WS) << std::scientific << std::right << "exchange"
       << std::setw(WS) << std::scientific << std::right << "grow-local"
       << std::setw(WS) << std::scientific << std::right << "grow-global"
       << std::endl;
    for (int i = 0; i < size; i++) {
      os << std::setw(5) << std::scientific << std::right << i
         << std::setw(WS) << std::scientific << std::right << tree_all[i]
         << std::setw(WS) << std::scientific << std::right << tree_sample[i]
         << std::setw(WS) << std::scientific << std::right << tree_exchange[i]
         << std::setw(WS) << std::scientific << std::right << tree_growlocal[i]
         << std::setw(WS) << std::scientific << std::right << tree_growglobal[i]
         << std::endl;
    }
    os << "----- end tree construction" << std::endl;
    os << std::endl;
  }

  V<double> let_all, let_trv, let_req, let_res, let_reg;
  tapas::mpi::Gather(data.time_let_all,      let_all, 0, comm);
  tapas::mpi::Gather(data.time_let_traverse, let_trv, 0, comm);
  tapas::mpi::Gather(data.time_let_req,      let_req, 0, comm);
  tapas::mpi::Gather(data.time_let_response, let_res, 0, comm);
  tapas::mpi::Gather(data.time_let_register, let_reg, 0, comm);
  
  if (rank == 0) {
    os << std::endl;
    os << "---- begin LET" << std::endl;
    os << std::setw(5) << std::scientific << std::right << "rank"
       << std::setw(WS) << std::scientific << std::right << "all"
       << std::setw(WS) << std::scientific << std::right << "Traversal"
       << std::setw(WS) << std::scientific << std::right << "Request"
       << std::setw(WS) << std::scientific << std::right << "Response"
       << std::setw(WS) << std::scientific << std::right << "Register"
       << std::endl;
    for (int i = 0; i < size; i++) {
      os << std::setw(5) << std::scientific << std::right << i
         << std::setw(WS) << std::scientific << std::right << let_all[i]
         << std::setw(WS) << std::scientific << std::right << let_trv[i]
         << std::setw(WS) << std::scientific << std::right << let_req[i]
         << std::setw(WS) << std::scientific << std::right << let_res[i]
         << std::setw(WS) << std::scientific << std::right << let_reg[i]
         << std::endl;
    }
    os << "----- end LET" << std::endl;
    os << std::endl;
  }

}

}
}

#endif // TAPAS_HOT_REPORT_H

