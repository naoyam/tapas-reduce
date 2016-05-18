/* vectormap_util.h -*- Coding: us-ascii-unix; -*- */
/* Copyright (C) 2015-2015 RIKEN AICS */

#ifndef TAPAS_VECTORMAP_UTIL_
#define TAPAS_VECTORMAP_UTIL_

/** @file directutil.h @brief Utils for direct part by Tesla. */

//#include "mpi.h"
//#include <vector>
//#include <string>
//#include <algorithm>
//#include <assert.h>

namespace tapas {

/*AHO*/ /* NOTE: BELOW BE A PROPER WAY TO KNOW THE USE OF MPI. */

#ifdef USE_MPI

/** Finds the number of MPI processes on a node, and returns
    rank-of-node, rank-in-node and nprocs-in-node.  It is collective
    for all ranks.  It uses MPI_Get_processor_name() to identify a
    node. */

static void rank_in_node(MPI_Comm comm, int &rankofnode, int &rankinnode,
                         int &nprocsinnode) {
  int nprocs, rank;
  int cc = MPI_Comm_size(comm, &nprocs);
  assert(cc == MPI_SUCCESS); (void)cc;
  cc = MPI_Comm_rank(comm, &rank);
  assert(cc == MPI_SUCCESS);
  int namelen;
  char name[MPI_MAX_PROCESSOR_NAME];
#if 1
  cc = MPI_Get_processor_name(name, &namelen);
  assert(cc == MPI_SUCCESS);
  name[MPI_MAX_PROCESSOR_NAME - 1] = 0;
#else
  /*TEST*/
  snprintf(name, MPI_MAX_PROCESSOR_NAME, "host%d", (rank / 3));
#endif

  char names[nprocs][MPI_MAX_PROCESSOR_NAME];
  cc = MPI_Allgather(name, MPI_MAX_PROCESSOR_NAME, MPI_BYTE,
                     names, MPI_MAX_PROCESSOR_NAME, MPI_BYTE, comm);
  assert(cc == MPI_SUCCESS);
  std::vector<int> group;
  std::string n (name);
  for (int i = 0; i < nprocs; i++) {
    if (std::string(names[i]) == n) {
      group.push_back(i);
    }
  }
  assert(group.size() > 0);
  std::vector<int>::iterator m = std::find(group.begin(), group.end(), rank);
  assert(m != group.end());

  int head = ((group[0] == rank) ? 1 : 0);
  int headrank;
  cc = MPI_Scan(&head, &headrank, 1, MPI_INT, MPI_SUM, comm);
  assert(cc == MPI_SUCCESS);

  int section [nprocs];
  for (int i = 0; i < nprocs; i++) {
    section[i] = 0;
  }
  if (head != 0) {
    for (std::vector<int>::iterator i = group.begin(); i != group.end(); i++) {
      section[(*i)] = (headrank - 1);
    }
  }

  int ranks [nprocs];
  cc = MPI_Allreduce(section, ranks, nprocs, MPI_INT, MPI_MAX, comm);
  assert(cc == MPI_SUCCESS);
  assert(!(head != 0) || ranks[rank] == (headrank - 1));
  int node = ranks[rank];

  for (int i = 0; i < nprocs; i++) {
    std::vector<int>::iterator g = std::find(group.begin(), group.end(), i);
    if (g == group.end()) {
      assert(ranks[i] != node);
    } else {
      assert(ranks[i] == node);
    }
  }

  std::vector<int>::iterator r = std::find(group.begin(), group.end(), rank);
  assert(r != group.end());
  int nth = (r - group.begin());
  assert(nth >= 0 && nth < (int)group.size());

  rankofnode = node;
  rankinnode = nth;
  nprocsinnode = (int)group.size();
}

#endif /* USE_MPI */


void SetGPU() {
  if (getenv("CUDA_VISIBLE_DEVICES") != nullptr) {
    // if CUDA_VISIBLE_DEVICES is already set, it seems CUDA device initialization is done.
    return;
  }

  int rank = 0;

#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

  int rankofnode = 0, rankinnode = 0, nprocsinnode = 1;
#ifdef USE_MPI
  rank_in_node(MPI_COMM_WORLD, rankofnode, rankinnode, nprocsinnode);
#endif

  std::stringstream ss;
  ss << rankinnode;
  setenv("CUDA_VISIBLE_DEVICES", ss.str().c_str(), 1);

  std::cout << "rank " << rank << " (rank in node = " << rankinnode << ", "
            << "rank of node = " << rankofnode << ") "
            << "CUDA_VISIBLE_DEVICES=" << getenv("CUDA_VISIBLE_DEVICES") << std::endl;

  // explicitly initialize CUDA runtime by calling dummy cudaMalloc().
  // Dummy CUDA API call to initialize the runtime (for performance measurement)
#ifdef __CUDACC__
  int *p = nullptr;
  cudaMalloc(&p, sizeof(int));
  cudaFree(p);
#endif
}

}

#endif /*TAPAS_VECTORMAP_UTIL_*/
