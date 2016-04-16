#ifndef TAPAS_HOT_REPORT_H
#define TAPAS_HOT_REPORT_H

#include <sys/unistd.h> // gethostname

#include <ostream>
#include <iostream>
#include <iomanip>
#include <algorithm>

#include "tapas/hot.h"
#include "tapas/util.h"

namespace tapas {
namespace hot {

using CSV = tapas::util::CSV;
using RankCSV = tapas::util::RankCSV;

namespace {
template<class T> using V = std::vector<T>;

struct HostName {
  static const constexpr int kLen = 200;
  char value[kLen];
};

void WriteHostName(const std::string &fname) {
  HostName hostname;
  gethostname(hostname.value, HostName::kLen);

  RankCSV writer {"Hostname"};
  writer.At("Hostname") = hostname.value;
  writer.Dump(fname);
}

}

template<class Data>
void Report(const Data &data) {
  // auto comm = data.mpi_comm_;
  auto comm = MPI_COMM_WORLD;

  int rank, size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  std::string report_prefix;
  std::string report_suffix;

  if (getenv("TAPAS_REPORT_PREFIX")) {
    report_prefix = getenv("TAPAS_REPORT_PREFIX");
  }

  if (getenv("TAPAS_REPORT_SUFFIX")) {
    report_suffix = getenv("TAPAS_REPORT_SUFFIX");
  }

  WriteHostName(report_prefix + "hostnames" + report_suffix + ".csv");

  // misc data
  if (rank == 0) {
    CSV csv({"SamplingRate", "NB"}, 1);
    csv.At("SamplingRate", 0) = data.sampling_rate;
    csv.At("NB", 0) = data.nb_total;
    csv.Dump(report_prefix + "misc" + report_suffix + ".csv");
  }

  // number of cells and bodies
  {
    RankCSV csv {"NumBodies1", "NumBodies2", "NumLeaves", "NumCells"};
    csv.At("NumBodies1") = data.nb_before;
    csv.At("NumBodies2") = data.nb_after;
    csv.At("NumLeaves")  = data.nleaves;
    csv.At("NumCells")   = data.ncells;

    csv.Dump(report_prefix + "load_balancing" + report_suffix + ".csv");
  }

  // Tree construction breakdown
  {
    RankCSV csv {"all", "sample", "exchange", "grow-local", "grow-global"};
    csv.At("all") = data.time_tree_all;
    csv.At("sample") = data.time_tree_sample;
    csv.At("exchange") = data.time_tree_exchange;
    csv.At("grow-local") = data.time_tree_growlocal;
    csv.At("grow-global") = data.time_tree_growglobal;
    csv.Dump(report_prefix + "tree_construction" + report_suffix + ".csv");
  }

  // LET exchange breakdown
  {
    RankCSV csv {"LET-All", "LET-Trav-main", "LET-Trav-sub",
          "LET-Req-all", "LET-Req-comm",
          "LET-Res-all", "LET-Res-acomm", "LET-Res-bcomm",
          "LET-Reg"};
    csv.At("LET-All") = data.time_let_all;
    csv.At("LET-Trav-main") = data.time_let_trav_main;
    csv.At("LET-Trav-sub") = data.time_let_trav_sub;
    csv.At("LET-Req-all") = data.time_let_req_all;
    csv.At("LET-Req-comm") = data.time_let_req_comm;
    csv.At("LET-Res-all") = data.time_let_res_all;
    csv.At("LET-Res-acomm")  = data.time_let_res_attr_comm;
    csv.At("LET-Res-bcomm") = data.time_let_res_body_comm;
    csv.At("LET-Reg") = data.time_let_register;
    csv.Dump(report_prefix + "let_construction" + report_suffix + ".csv");
  }

#if 0
  // Report how many keys are used from ht_let_
  tapas::debug::BarrierExec([&data](int rank, int) {
      if (rank == 0) {
        std::cout << "LET density" << std::endl;
      }

      int total = data.ht_let_.size();
      int used = 0;
      for (auto k : data.trav_used_src_key_) {
        if (data.ht_let_.count(k) > 0) {
          used++;
        }
      }
      //int used  = data.let_used_key_.size();
      //double r = used * 1.0 / total;
      std::cout << "Rank " << rank << ": Used = " << used << " / "
                << total
                << std::endl;
    });
#endif

#if 0
  if (data.mpi_rank_ == 0) {
    std::cout << "Max Depth = " << data.max_depth_ << std::endl;
  }
#endif

  // Map2 breakdown
  {
    RankCSV csv {"all", "let", "net_traverse"
#ifdef __CUDACC__
          , "device_call"
#endif
          };
    csv.At("all") = data.time_map2_all;
    csv.At("let") = data.time_map2_let;
    csv.At("net_traverse") = data.time_map2_net;
#ifdef __CUDACC__
    csv.At("device_call") = data.time_map2_dev;
#endif
    csv.Dump(report_prefix + "map2" + report_suffix + ".csv");
  }
}

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_REPORT_H
