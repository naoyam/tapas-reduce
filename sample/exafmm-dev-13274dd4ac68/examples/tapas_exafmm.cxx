
#include <chrono>
#include <mutex>
#include <thread>

#include <sys/types.h>
#include <unistd.h> // usleep

#ifdef USE_MPI
# include <mpi.h>
#endif

// NOTE:
// This macro (TO_MTHREAD_NATIVE or TO_SERIAL) is needed by tpswitch.h, which included in the original ExaFMM.
// Although this is not necessary in Tapas, ExaFMM's logger class is still using it.
#if MTHREAD
# define TO_MTHREAD_NATIVE 1
# define TO_SERIAL 0
#else /* MTHREAD */
# define TO_SERIAL 1
#endif

#include "args.h"
#include "dataset.h"
#include "logger.h"
#include "kernel.h"
//#include "up_down_pass.h"
#include "verify.h"

#include "tapas_exafmm.h"
#include "LaplaceSphericalCPU_tapas.h"
#include "LaplaceP2PCPU_tapas.h"

#ifdef TBB
# include <tbb/task_scheduler_init.h>
#endif

#ifdef COUNT /* Count kernel invocations */

# warning "COUNT is defined. This may significantly slows down execution"
uint64_t numM2L = 0;
uint64_t numP2P = 0;
inline void ResetCount() { numP2P = 0; numM2L = 0; }

#else

inline void ResetCount() { }

#endif /* ifdef COUNT */

#ifdef USE_RDTSC
# ifdef TAPAS_COMPILER_INTEL
#  define RDTSC() __rdtsc()
# endif
#else
# define RDTSC() 0
#endif

double GetTime() {
#ifdef USE_MPI
  return MPI_Wtime();
#else
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec + tv.tv_usec * 1e-6;
#endif
}

template <int DIM, class FP> inline
tapas::Vec<DIM, FP> &asn(tapas::Vec<DIM, FP> &dst, const vec<DIM, FP> &src) {
  for (int i = 0; i < DIM; ++i) {
    dst[i] = src[i];
  }
  return dst;
}

template <int DIM, class FP> inline
vec<DIM, FP> &asn(vec<DIM, FP> &dst, const tapas::Vec<DIM, FP> &src) {
  for (int i = 0; i < DIM; ++i) {
    dst[i] = src[i];
  }
  return dst;
}

// UpDownPass::upwardPass
struct FMM_Upward {
  template<class Cell>
  inline void operator()(Cell &c, real_t theta) {
    // theta is not used now; to be deleted
    auto attr = c.attr();
    attr.R = 0;
    attr.M = 0;
    attr.L = 0;
    c.attr() = attr;

#ifdef TAPAS_DEBUG_DUMP
    {
      tapas::debug::DebugStream e("FMM_Upward");
      e.out() << TapasFMM::TSP::SFC::Simplify(c.key()) << " (1) " << c.IsLeaf() << " ";
      e.out() << "c.attr().R = " << std::fixed << std::setprecision(6) << c.attr().R << " ";
      e.out() << std::endl;
    }
#endif

    if (c.IsLeaf()) {
      P2M(c);
    } else {
      tapas::Map(*this, c.subcells(), theta);
      M2M(c);
    }

#if 0 // to be removed
    for (int i = 0; i < 3; ++i) {
      c.attr().R = std::max(c.width(i), c.attr().R);
    }

    c.attr().R = c.attr().R / 2 * 1.00001; // see bounds2box func
    c.attr().R /= theta;
#endif
  }
};

struct FMM_Downward {
  template<class Cell>
  inline void operator()(Cell &c) {
    //if (c.nb() == 0) return;
    if (!c.IsRoot()) {
      L2L(c);
    }
    
    if (c.IsLeaf()) {
      if (c.nb() > 0) {
        tapas::Map(L2P, c.bodies(), &c);
      }
    } else {
      tapas::Map(*this, c.subcells());
    }
  }
};

#define RANK 0
#define KEY 2522015791327477762

template<class Cell, class L>
void Debug(Cell &Ci, Cell &Cj, L lambda) {
  int rank = 0;
  (void) Ci;
  (void) Cj;
#ifdef USE_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
  if (getenv("TAPAS_IN_LET")) {
    lambda(rank);
  }
}


#ifdef TAPAS_DEBUG

template<class Cell>
void DebugWatchCell(Cell &Ci, Cell &Cj, real_t Ri, real_t Rj, real_t R2) {
  using KeyType = typename Cell::SFC::KeyType;
  const KeyType C4 = 221802281647996932; // the problematic cell
  const KeyType C3 = 216172782113783811;
  const KeyType C2 = 216172782113783811;
  const KeyType C1 = 1;

  const KeyType watched_cell = C3;
  const int watched_rank = 1;

  using SFC = typename Cell::SFC;

  if (tapas::debug::MPI_Rank() == watched_rank) {
    if (Cj.key() == watched_cell) {
      if (getenv("TAPAS_IN_LET")) {
        std::cout << "In LET: Found src_key = " << Cj.key() << std::endl;
      } else {
        std::cout << "In Main Traverse: Found src_key = " << Cj.key() << std::endl;
      }
      std::cout << "\tCi.key() = " << SFC::Describe(Ci.key()) << std::endl;
      std::cout << "\tCj.key() = " << SFC::Describe(Cj.key()) << std::endl;
      std::cout << "\tRi = " << Ri << std::endl;
      std::cout << "\tRj = " << Rj << std::endl;
      std::cout << "\tR2 = " << R2 << std::endl;
      std::cout << "\t(Ri + Rj) * (Ri + Rj) = " << ((Ri + Rj) * (Ri + Rj)) << std::endl;
      std::cout << "\t" << "target width  = " << Ci.region().width() << std::endl;
      std::cout << "\t" << "target center = " << Ci.region().center() << std::endl;
      std::cout << "\t" << "source width  = " << Cj.region().width() << std::endl;
      std::cout << "\t" << "source center = " << Cj.region().center() << std::endl;
      if (R2 > (Ri + Rj) * (Ri + Rj)) {                   // If distance is far enough
        std::cout << "\tApprox" << std::endl;
      } else if (Ci.IsLeaf() && Cj.IsLeaf()) {            // Else if both cells are bodies
        std::cout << "\tP2P" << std::endl;
      } else {                                                    // Else if cells are close but not bodies
        std::cout << "\tSplit" << ", "
                  << "Ci.IsLeaf() = " << Ci.IsLeaf() << ", "
                  << "Cj.IsLeaf() = " << Cj.IsLeaf() << std::endl;
      }                                                           // End if for multipole acceptance
      std::cout << std::endl;
    }
  }
}

#endif


// Perform ExaFMM's Dual Tree Traversal (M2L & P2P)
struct FMM_DTT {
  template<class Cell>
  inline void operator()(Cell &Ci, Cell &Cj, int mutual, int nspawn, real_t theta) {

// #ifdef COUNT
//     if (Ci.IsRoot() && Cj.IsRoot()) { // ad-hoc
//       ResetCount();
//     }
// #endif

    // TODO:
    //if (Ci.nb() == 0 || Cj.nb() == 0) return;

    //real_t R2 = (Ci.center() - Cj.center()).norm();
    real_t R2 = Ci.Distance(Cj, tapas::Center);
    vec3 Xperiodic = 0; // dummy; periodic is not ported

    real_t Ri = 0;
    real_t Rj = 0;

    for (int d = 0; d < 3; d++) {
      Ri = std::max(Ri, Ci.width(d));
      Rj = std::max(Rj, Cj.width(d));
    }

    Ri = (Ri / 2 * 1.00001) / theta;
    Rj = (Rj / 2 * 1.00001) / theta;

    //DebugWatchCell(Ci, Cj, Ri, Rj, R2);

    if (R2 > (Ri + Rj) * (Ri + Rj)) {                   // If distance is far enough
      // tapas::Apply(M2L, Ci, Cj, Xperiodic, mutual); // \todo
      // if (!Cell::Inspector) M2L(Ci, Cj, Xperiodic, mutual);
      M2L(Ci, Cj, Xperiodic, mutual);                   //  M2L kernel
      
    } else if (Ci.IsLeaf() && Cj.IsLeaf()) {            // Else if both cells are bodies
      tapas::Map(P2P(), tapas::Product(Ci.bodies(), Cj.bodies()), Xperiodic, mutual);
    } else {                                                    // Else if cells are close but not bodies
      tapas_splitCell(Ci, Cj, Ri, Rj, mutual, nspawn, theta);   //  Split cell and call function recursively for child
    }                                                           // End if for multipole acceptance
  }

  template<class Cell>
  inline void tapas_splitCell(Cell &Ci, Cell &Cj, real_t Ri, real_t Rj, int mutual, int nspawn, real_t theta) {
    (void) Ri; (void) Rj;
    bool Ci_IsLeaf = Ci.IsLeaf();
    if (Cj.IsLeaf()) {
      assert(!Ci.IsLeaf());                                   //  Make sure Ci is not leaf
      //for (C_iter ci=Ci0+Ci->ICHILD; ci!=Ci0+Ci->ICHILD+Ci->NCHILD; ci++) {// Loop over Ci's children
      //  traverse(ci, Cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
      //}                                                         //
      //End loop over Ci's children
      tapas::Map(*this, tapas::Product(Ci.subcells(), Cj), mutual, nspawn, theta);
    } else if (Ci_IsLeaf) {                               // Else if Ci is leaf
      //} else if (Ci.IsLeaf()) {                               // Else if Ci is leaf
      assert(!Cj.IsLeaf());                                   //  Make sure Cj is not leaf
      //for (C_iter cj=Cj0+Cj->ICHILD; cj!=Cj0+Cj->ICHILD+Cj->NCHILD; cj++) {// Loop over Cj's children
      //  traverse(Ci, cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
      //}                                                         //
      //End loop over Cj's children
      tapas::Map(*this, tapas::Product(Ci, Cj.subcells()), mutual, nspawn, theta);
    } else if (mutual && Ci == Cj) {
      tapas::Map(*this, tapas::Product(Ci.subcells(), Cj.subcells()), mutual, nspawn, theta);
#if 0
    // } else if (Ci.local_nb() >= nspawn) {
    //   //split both
    //   tapas::Map(*this, tapas::Product(Ci.subcells(), Cj.subcells()), mutual, nspawn, theta);
    } else if (Ri >= Rj) {                                // Else if Ci is larger than Cj
      // split target(left)
      tapas::Map(*this, tapas::Product(Ci.subcells(), Cj), mutual, nspawn, theta);
    } else {                                                    // Else if Cj is larger than Ci
      // split source(right)
      tapas::Map(*this, tapas::Product(Ci, Cj.subcells()), mutual, nspawn, theta);
    }
#else
    } else {
      // Split both side
      tapas::Map(*this, tapas::Product(Ci.subcells(), Cj.subcells()), mutual, nspawn, theta);
  }
#endif
  }
};

void CheckResult(Bodies &bodies, int numSamples, real_t cycle, int images) {

  int mpi_rank = 0;
  int mpi_size = 1;

#ifdef USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
#endif

  numSamples = std::min(numSamples, (int)bodies.size());

  Bodies targets(numSamples);
  Bodies samples(numSamples);

  int stride = bodies.size() / numSamples;

  if (mpi_rank == 0) {
    for (int i=0, j=0; i < numSamples; i++,j+=stride) {
      samples[i] = bodies[j];
      targets[i] = bodies[j];
    }
    Dataset().initTarget(samples);
  }

  int prange = 0;
  for (int i=0; i<images; i++) {
    prange += int(std::pow(3.,i));
  }

  for (int p = 0; p < mpi_size; p++) {
    if (p == mpi_rank) {
      std::cout << "Computing on rank " << p << " against " << bodies.size() << " bodies." << std::endl;
      Cells cells;
      cells.resize(2);
      C_iter Ci = cells.begin(), Cj = cells.begin() + 1;
      Ci->BODY  = samples.begin();
      Ci->NBODY = samples.size();
      Cj->BODY  = bodies.begin();
      Cj->NBODY = bodies.size();

      vec3 Xperiodic = 0;
      for (int ix=-prange; ix<=prange; ix++) {
        for (int iy=-prange; iy<=prange; iy++) {
          for (int iz=-prange; iz<=prange; iz++) {
            Xperiodic[0] = ix * cycle;
            Xperiodic[1] = iy * cycle;
            Xperiodic[2] = iz * cycle;
            kernel::P2P(Ci, Cj, Xperiodic, false);
          }
        }
      }
    }

#ifdef USE_MPI
    // Send sampled bodies to rank p to rank (p+1) % mpi_size
    int src = p;
    int dst = (p + 1) % mpi_size;

    if (src != dst) {
      if (src == mpi_rank) {
        MPI_Send(samples.data(), sizeof(samples[0]) * samples.size(), MPI_BYTE, dst, 0, MPI_COMM_WORLD);
      } else if (dst == mpi_rank) {
        MPI_Status stat;
        MPI_Recv(samples.data(), sizeof(samples[0]) * samples.size(), MPI_BYTE, src, 0, MPI_COMM_WORLD, &stat);
      }
    }
#endif
  }

  // Traversal::normalize()
  for (auto b = samples.begin(); b != samples.end(); b++) {
    b->TRG /= b->SRC;
  }

  Verify verify;
  double potDif = verify.getDifScalar(samples, targets);
  double potNrm = verify.getNrmScalar(samples);
  double accDif = verify.getDifVector(samples, targets);
  double accNrm = verify.getNrmVector(samples);

  logger::printTitle("FMM vs. direct");
  // std::cout << "potDif = " << potDif << std::endl;
  // std::cout << "potNrm = " << potDif << std::endl;
  // std::cout << "accDif = " << potDif << std::endl;
  // std::cout << "accNrm = " << potDif << std::endl;
  verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
  verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
  std::cout.flush();
}

/**
 * \brief Copy particle informations from Tapas to user's program to check result
 */
static inline void CopyBackResult(Bodies &bodies, TapasFMM::Cell *root) {
  bodies.clear();

  Body *beg = &root->local_body(0);
  Body *end = beg + root->local_nb();
  bodies.assign(beg, end); // assign body attributes

  for (size_t i = 0; i < bodies.size(); i++) {
    bodies[i].TRG = root->local_body_attr(i);
  }
}


std::string Now() {
  time_t now = time(NULL);
  struct tm *pnow = localtime(&now);

  std::stringstream ss;
  ss << pnow->tm_year + 1900 << "/"
     << (pnow->tm_mon + 1) << "/"
     << pnow->tm_mday << " "
     << pnow->tm_hour << ":"
     << pnow->tm_min << ":"
     << pnow->tm_sec;
  return ss.str();
}

void PrintProcInfo() {
  const constexpr int HOSTNAME_LEN = 20;
  char hostname[HOSTNAME_LEN];
  gethostname(hostname, HOSTNAME_LEN);

  int pid = getpid();

#ifdef USE_MPI
  int rank = 0;
  int size = 1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char *rbuf_hn = (rank == 0) ? new char[HOSTNAME_LEN * size] : nullptr;
  MPI_Gather(hostname, HOSTNAME_LEN, MPI_BYTE, rbuf_hn, HOSTNAME_LEN, MPI_BYTE, 0, MPI_COMM_WORLD);

  int *rbuf_pid = (rank == 0) ? new int[size] : nullptr;
  MPI_Gather(&pid, 1, MPI_INT, rbuf_pid, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      std::string hn(rbuf_hn + HOSTNAME_LEN * i);
      std::cout << "MPI Rank " << i << " " << hn << ":" << rbuf_pid[i] << std::endl;
    }
  }

  sleep(10); // for gdb attach
  MPI_Barrier(MPI_COMM_WORLD);
#else
  std::cout << "MPI Rank 0" << hostname << ":" << pid << std::endl;
#endif
}

int main(int argc, char ** argv) {
  Args args(argc, argv);

#ifdef USE_MPI
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &args.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &args.mpi_size);

#endif

#if 0
  PrintProcInfo();
#endif

#ifdef TBB
  if (TBB_INTERFACE_VERSION != TBB_runtime_interface_version()) {
    if (args.mpi_rank == 0) {
      std::cerr << "Compile-time and run-time TBB versions do not match." << std::endl;
    }
    abort();
  }

  if (args.mpi_rank == 0) {
    std::cout << "TBB: version = " << TBB_runtime_interface_version() << std::endl;
    std::cout << "TBB: Initializing threads = " << args.threads << std::endl;
  }
  task_scheduler_init init(args.threads);
#endif

  // This function is called by Tapas automatically and user program doesn't need to call it.
  // In this case, however, we want to exclude initialization cost of CUDA runtime from performance
  // measurement.
  tapas::SetGPU();
  
  if (args.mpi_rank == 0) {
    std::cout << "Threading model " << FMM_Threading::name() << std::endl;
  }

  // ad-hoc code for MassiveThreads when used with mvapich.
#ifdef MTHREADS
  FMM_Threading::init();
#endif

  Bodies bodies, bodies2, bodies3, jbodies;
  Cells cells, jcells;
  Dataset data;

  if (args.useRmax) {
    std::cerr << "Rmax not supported." << std::endl;
    std::cerr << "Use --useRmax 0 option." << std::endl;
    exit(1);
  }
  if (args.useRopt) {
    std::cerr << "Ropt not supported." << std::endl;
    std::cerr << "Use --useRopt 0 option." << std::endl;
    exit(1);
  }

#ifdef __CUDACC__
  if (args.mutual) {
    std::cerr << "TapasFMM: [Error] Mutual is not supported for CUDA implementation in this version." << std::endl;
#ifdef USE_MPI
    MPI_Finalize();
#endif
    exit(-1);
  }
#endif

  //UpDownPass upDownPass(args.theta, args.useRmax, args.useRopt);
  Verify verify;
  (void) verify;

  Region tr;

  logger::startTimer("Dataset generation");
  bodies = data.initBodies(args.numBodies, args.distribution, args.mpi_rank, args.mpi_size);
  logger::stopTimer("Dataset generation");

  // Dump all bodies data for debugging
#ifdef TAPAS_DEBUG_DUMP
  {
    tapas::debug::DebugStream err("bodies");
    for (auto &b : bodies) {
      err.out() << b.X << " " << b.SRC << std::endl;
    }
  }
#endif

  const real_t cycle = 2 * M_PI;
  logger::verbose = args.verbose && (args.mpi_rank == 0);
  if (args.mpi_rank == 0) {
    logger::printTitle("FMM Parameters");
    args.print(logger::stringLength, P);
  }

  double time_upw = 0, time_dtt = 0, time_dwn = 0, time_tree = 0;

  if (args.mpi_rank == 0) {
    std::cout << "Starting FMM timesteps" << std::endl;
  }

  for (int t=0; t<args.repeat; t++) {
    double total_bt = GetTime();
    logger::printTitle("FMM Profiling");
    logger::startTimer("Total FMM");
    logger::startPAPI();
    logger::startDAG();

    TapasFMM::Cell *root = nullptr;
    {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      double bt = GetTime();

      root = TapasFMM::Partition(bodies.data(), bodies.size(), args.ncrit);

      double et = GetTime();
      time_tree = et - bt;
    }

    root->SetOptMutual(args.mutual);

#ifdef USE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif

    // Upward (P2M + M2M)
    {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      logger::startTimer("Upward pass");
      double bt = GetTime();

      tapas::Map(FMM_Upward(), *root, args.theta);

      double et = GetTime();
      logger::stopTimer("Upward pass");

      time_upw = et - bt;
    }

#ifdef TAPAS_DEBUG_DUMP
    dumpM(*root);
#endif

    {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      logger::startTimer("Traverse");
      double bt = GetTime();
      ResetCount();

      tapas::Map(FMM_DTT(), tapas::Product(*root, *root), args.mutual, args.nspawn, args.theta);

      double et = GetTime();
      logger::stopTimer("Traverse");
      time_dtt = et - bt;
    }

    TAPAS_LOG_DEBUG() << "Dual Tree Traversal done\n";
    jbodies = bodies;

#ifdef TAPAS_DEBUG_DUMP
    dumpL(*root);
#endif

    {
#ifdef USE_MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      logger::startTimer("Downward pass");
      double bt = GetTime();

      tapas::Map(FMM_Downward(), *root);
      //tapas::DownwardMap(FMM_Downward(), *root);

      double et = GetTime();
      logger::stopTimer("Downward pass");
      time_dwn = et - bt;
    }

    TAPAS_LOG_DEBUG() << "L2P done\n";

#ifdef TAPAS_DEBUG_DUMP
    dumpBodies(*root);
#endif

    CopyBackResult(bodies, root);
    //CopyBackResult(bodies, root->body_attrs(), args.numBodies);

    double total_et = GetTime();
    logger::printTitle("Total runtime");
    logger::stopPAPI();
    logger::stopTimer("Total FMM");
    logger::resetTimer("Total FMM");

    double time_total = total_et - total_bt;
    tapas::util::RankCSV csv {"total", "upward", "traverse", "downward", "tree"
#ifdef COUNT
          , "numP2P", "numM2L"
#endif
          };
    csv.At("total") = time_total;
    csv.At("upward") = time_upw;
    csv.At("traverse") = time_dtt;
    csv.At("downward") = time_dwn;
    csv.At("tree") = time_tree;
#ifdef COUNT
    csv.At("numP2P") = numP2P;
    csv.At("numM2L") = numM2L;
#endif

    std::string report_prefix;
    std::string report_suffix;

    if (getenv("TAPAS_REPORT_PREFIX")) {
      report_prefix = getenv("TAPAS_REPORT_PREFIX");
    }

    if (getenv("TAPAS_REPORT_SUFFIX")) {
      report_suffix = getenv("TAPAS_REPORT_SUFFIX");
    }

    csv.Dump(report_prefix + "main" + report_suffix + ".csv");

#if WRITE_TIME
    logger::writeTime();
#endif

    if (args.check) {
      const int numTargets = 100;
      logger::startTimer("Total Direct");
      CheckResult(bodies, numTargets, cycle, args.images);
      logger::stopTimer("Total Direct");
    }

    //buildTree.printTreeData(cells);
    logger::printPAPI();
    logger::stopDAG();

#ifdef COUNT
    if (args.mpi_rank == 0) {
      std::cout << "P2P calls" << " : " << numP2P << std::endl;
      std::cout << "M2L calls" << " : " << numM2L << std::endl;
    }
#endif

    bodies = bodies3;
    data.initTarget(bodies);

    root->Report();
  } /* end for */

#ifdef USE_MPI
  MPI_Finalize();
#endif

  return 0;
}
