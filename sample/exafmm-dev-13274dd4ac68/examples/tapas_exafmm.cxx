#include <thread>
#include <mutex>
#include <unistd.h> // usleep

#ifdef USE_MPI
# include <mpi.h>
#endif

// NOTE:
// This macro (TO_MTHREAD_NATIVE or TO_SERIAL) is needed by tpswitch.h, which included in the original ExaFMM.
// Although this is not necessary in Tapas, ExaFMM's BoundBox class is still used just for computing bounding boxes.
#if MTHREADS
# define TO_MTHREAD_NATIVE
#else /* MTHREADS */
# define TO_SERIAL
#endif

#include "args.h"
#include "bound_box.h"
#ifdef CILK
#include "build_tree3.h"
#else
#include "build_tree.h"
#endif
#include "dataset.h"
#include "logger.h"
#include "up_down_pass.h"
#include "verify.h"

#include "tapas_exafmm.h"
#include "LaplaceSphericalCPU_tapas.h"
#include "LaplaceP2PCPU_tapas.h"

#ifdef TAPAS_USE_VECTORMAP
#  include "LaplaceP2PCPU_tapas.cxx"
#endif /*TAPAS_USE_VECTORMAP*/

int numM2L = 0;
int numP2P = 0;

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

static Region &asn(Region &x, const Bounds &y) {
  asn(x.min(), y.Xmin);
  asn(x.max(), y.Xmax);
  return x;
}

// UpDownPass::upwardPass
static inline void FMM_Upward(Tapas::Cell &c, real_t theta) {
  c.attr().R = 0;
  c.attr().M = 0;
  c.attr().L = 0;
  
#ifdef TAPAS_DEBUG
  {
    tapas::debug::DebugStream e("FMM_Upward");
    e.out() << Tapas::SFC::Simplify(c.key()) << " (1) " << c.IsLeaf() << " ";
    e.out() << "c.attr().R = " << std::fixed << std::setprecision(6) << c.attr().R << " ";
    e.out() << std::endl;
  }
#endif
  
  if (c.IsLeaf()) {
    P2M(c);
  } else {
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

static inline void FMM_Downward(Tapas::Cell &c) {
  //if (c.nb() == 0) return;
  if (!c.IsRoot()) L2L(c);
  if (c.IsLeaf() && c.nb() > 0) {
    tapas::Map(L2P, c.bodies());
  }
}

// Perform ExaFMM's Dual Tree Traversal (M2L & P2P)
struct FMM_DTT {
  template<class Cell>
  inline void operator()(Cell &Ci, Cell &Cj, int mutual, int nspawn, real_t theta) {
    // TODO:
    //if (Ci.nb() == 0 || Cj.nb() == 0) return;

    vec3 dX;
    asn(dX, Ci.center() - Cj.center());
    real_t R2 = norm(dX);
    vec3 Xperiodic = 0; // dummy; periodic not ported

    real_t Ri = 0;
    real_t Rj = 0;

    for (int d = 0; d < 3; d++) {
      Ri = std::max(Ri, Ci.width(d));
      Rj = std::max(Rj, Cj.width(d));
    }
    Ri = (Ri / 2 * 1.00001) / theta;
    Rj = (Rj / 2 * 1.00001) / theta;
    
    if (R2 > (Ri + Rj) * (Ri + Rj)) {                   // If distance is far enough
      numM2L++;
      M2L(Ci, Cj, Xperiodic, mutual);                   //  M2L kernel
    } else if (Ci.IsLeaf() && Cj.IsLeaf()) {            // Else if both cells are bodies
#ifdef TAPAS_USE_VECTORMAP
      tapas::Map(P2P(), tapas::Product(Ci.bodies(), Cj.bodies()), Xperiodic, mutual);
#else
      tapas::Map(P2P(), tapas::Product(Ci.bodies(), Cj.bodies()), Xperiodic, mutual);
#endif /*TAPAS_USE_VECTORMAP*/

      numP2P++;
    } else {                                                    // Else if cells are close but not bodies
      tapas_splitCell(Ci, Cj, Ri, Rj, mutual, nspawn, theta);   //  Split cell and call function recursively for child
    }                                                           // End if for multipole acceptance
  }
  
  template<class Cell>
  inline void tapas_splitCell(Cell &Ci, Cell &Cj, real_t Ri, real_t Rj, int mutual, int nspawn, real_t theta) {
    if (Cj.IsLeaf()) {
      assert(!Ci.IsLeaf());                                   //  Make sure Ci is not leaf
      //for (C_iter ci=Ci0+Ci->ICHILD; ci!=Ci0+Ci->ICHILD+Ci->NCHILD; ci++) {// Loop over Ci's children
      //  traverse(ci, Cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
      //}                                                         //
      //End loop over Ci's children
      tapas::Map(*this, tapas::Product(Ci.subcells(), Cj), mutual, nspawn, theta);
    } else if (Ci.IsLeaf()) {                               // Else if Ci is leaf
      assert(!Cj.IsLeaf());                                   //  Make sure Cj is not leaf
      //for (C_iter cj=Cj0+Cj->ICHILD; cj!=Cj0+Cj->ICHILD+Cj->NCHILD; cj++) {// Loop over Cj's children
      //  traverse(Ci, cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
      //}                                                         //
      //End loop over Cj's children
      tapas::Map(*this, tapas::Product(Ci, Cj.subcells()), mutual, nspawn, theta);
    } else if (mutual && Ci == Cj) {
      tapas::Map(*this, tapas::Product(Ci.subcells(), Cj.subcells()), mutual, nspawn, theta);
#if 0
    } else if (Ci.nb() + Cj.nb() >= (tapas::index_t)nspawn || (mutual && Ci == Cj)) {// Else if cells are still large
      //TraverseRange traverseRange(this, Ci0+Ci->ICHILD, Ci0+Ci->ICHILD+Ci->NCHILD,// Instantiate recursive functor
      //Cj0+Cj->ICHILD, Cj0+Cj->ICHILD+Cj->NCHILD, Xperiodic, mutual, remote);
      //traverseRange();                                          //
      //Traverse for range of cell pairs
      tapas::Map(*this, tapas::Product(Ci.subcells(), Cj.subcells()), mutual, nspawn, theta);
#endif
    } else if (Ri >= Rj) {                                // Else if Ci is larger than Cj
      //for (C_iter ci=Ci0+Ci->ICHILD; ci!=Ci0+Ci->ICHILD+Ci->NCHILD; ci++) {// Loop over Ci's children
      //  traverse(ci, Cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
      //}                                                         //  End loop over Ci's children
      tapas::Map(*this, tapas::Product(Ci.subcells(), Cj), mutual, nspawn, theta);
    } else {                                                    // Else if Cj is larger than Ci
      //for (C_iter cj=Cj0+Cj->ICHILD; cj!=Cj0+Cj->ICHILD+Cj->NCHILD; cj++) {// Loop over Cj's children
      //  traverse(Ci, cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
      //}                                                         //  End loop over Cj's children
      tapas::Map(*this, tapas::Product(Ci, Cj.subcells()), mutual, nspawn, theta);
    }
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
      std::cerr << "Computing on rank " << p << " against " << bodies.size() << " bodies." << std::endl;
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
}

/**
 * \brief Copy particle informations from Tapas to user's program to check result
 */
static inline void CopyBackResult(Bodies &bodies, Tapas::Cell *root) {
  bodies.clear();

  Body *beg = &root->local_body(0);
  Body *end = beg + root->local_nb();
  bodies.assign(beg, end); // assign body attributes

  for (size_t i = 0; i < bodies.size(); i++) {
    bodies[i].TRG = root->local_body_attr(i);
  }
}


// Debug function: Dump the M vectors of all cells.
void dumpM(Tapas::Cell &root) {
  std::stringstream ss;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << "M." << std::setw(4) << std::setfill('0') << rank << ".dat";
#else
  ss << "M.dat";
#endif
  std::mutex mtx;
  std::ofstream ofs(ss.str().c_str());
  std::function<void(Tapas::Cell&)> dump = [&](Tapas::Cell& cell) {
    mtx.lock();
    ofs << std::setw(20) << std::right << Tapas::SFC::Simplify(cell.key()) << " ";
    ofs << std::setw(3) << cell.depth() << " ";
    ofs << (cell.IsLeaf() ? "L" : "_") << " ";
    ofs << cell.attr().M << std::endl;
    mtx.unlock();
  };
  tapas::UpwardMap(dump, root);
  ofs.close();
}

// Debug function: Dump the L vectors of all cells.
void dumpL(Tapas::Cell &root) {
  std::stringstream ss;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << "L." << std::setw(4) << std::setfill('0') << rank << ".dat";
#else
  ss << "L.dat";
#endif
  std::mutex mtx;
  std::ofstream ofs(ss.str().c_str());
  std::function<void(Tapas::Cell&)> dump = [&](Tapas::Cell& cell) {
    mtx.lock();
    ofs << std::setw(20) << std::right << cell.key() << " ";
    ofs << std::setw(3) << std::noshowpos << cell.depth() << " ";
    ofs << (cell.IsLeaf() ? "L" : "_") << " ";

    for (size_t i = 0; i < cell.attr().L.size(); i++) {
      auto v = cell.attr().L[i];
      double real = v.real();
      double imag = v.imag();

      if (-1e-8 < real && real < 0) real = +0.0;
      if (-1e-8 < imag && imag < 0) imag = +0.0;
      
      ofs << "("
          << std::fixed << std::setprecision(5) << std::showpos << real << ","
          << std::fixed << std::setprecision(5) << std::showpos << imag
          << ") ";
    }
    //ofs << cell.attr().L << std::endl;
    ofs << std::endl;
    mtx.unlock();
  };
  tapas::UpwardMap(dump, root);
  ofs.close();
}

// Debug function: Dump the body attrs of all cells
void dumpBodies(Tapas::Cell &root) {
  std::stringstream ss;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << "bodies." << std::setw(4) << std::setfill('0') << rank << ".dat";
#else
  ss << "bodies.dat";
#endif
  std::mutex mtx;
  std::ofstream ofs(ss.str().c_str());
  std::function<void(Tapas::Cell&)> dump = [&](Tapas::Cell& cell) {
    if (cell.IsLeaf()) {
      mtx.lock();
      //ofs << std::setw(20) << std::right << Tapas::SFC::Simplify(cell.key()) << " ";
      auto iter = cell.bodies();
      for (int bi=0; bi < cell.nb(); bi++, iter++) {
        ofs << std::showpos << iter->X << " ";
        ofs << std::showpos << iter->SRC << " " << "vec4= ";
        for (int j = 0; j < 4; j++) {
          ofs << std::right
              << std::setiosflags(std::ios::showpos)
              << iter.attr()[j] << " ";
        }
        ofs << " " << cell.key() << std::endl;
      }
      mtx.unlock();
    }
  };
  tapas::UpwardMap(dump, root);
  ofs.close();
}

// Debug function: dump all leaves
void dumpLeaves(Tapas::Cell &root) {
  std::stringstream ss;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << "leaves." << std::setw(4) << std::setfill('0') << rank << ".dat";
#else
  ss << "leaves.dat";
#endif
  std::mutex mtx;
  std::ofstream ofs(ss.str().c_str(), std::ios_base::app);
  
  std::function<void(Tapas::Cell&)> f = [&](Tapas::Cell& cell) {
    if (cell.IsLeaf()) {
      mtx.lock();
      ofs << std::setw(20) << cell.key() << ", depth=" << cell.depth() << ", nb=" << cell.nb() << ", r=" << cell.region() << std::endl;
      for (int i = 0; i < cell.nb(); i++) {
        ofs << "    body[" << i << "]=(" << cell.body(i).X << ") " << std::endl;
      }
      mtx.unlock();
    } else {
      tapas::Map(f, cell.subcells());
    }
  };

  tapas::Map(f, root);
  
  ofs.close();
}

int main(int argc, char ** argv) {
  Args args(argc, argv);
  
#ifdef USE_MPI
  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &args.mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &args.mpi_size);
#endif

#ifdef TAPAS_USE_VECTORMAP
  Tapas::Cell::TSPClass::Vectormap::vectormap_setup(64, 31);
#endif /*TAPAS_USE_VECTORMAP*/

  Bodies bodies, bodies2, bodies3, jbodies;
  BoundBox boundBox(args.nspawn);
  Bounds bounds;
  BuildTree buildTree(args.ncrit, args.nspawn);
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
  //UpDownPass upDownPass(args.theta, args.useRmax, args.useRopt);
  Verify verify;
  (void) verify;

  Region tr;

  num_threads(args.threads);

  logger::startTimer("Dataset generation");
  bodies = data.initBodies(args.numBodies, args.distribution, args.mpi_rank, args.mpi_size);
  logger::stopTimer("Dataset generation");

  // Dump all bodies data for debugging
#ifdef TAPAS_DEBUG
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
  
  for (int t=0; t<args.repeat; t++) {
    logger::printTitle("FMM Profiling");
    logger::startTimer("Total FMM");
    logger::startPAPI();
    logger::startDAG();
    bounds = boundBox.getBounds(bodies);
    
    asn(tr, bounds);
    TAPAS_LOG_DEBUG() << "Bounding box: " << tr << std::endl;

    Tapas::Cell *root = Tapas::Partition(bodies.data(), bodies.size(), tr, args.ncrit);

    root->SetOptMutual(args.mutual);
    
#ifdef USE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    logger::startTimer("Upward pass");
    tapas::UpwardMap(FMM_Upward, *root, args.theta);
    logger::stopTimer("Upward pass");
    
#ifdef TAPAS_DEBUG
    dumpM(*root);
#endif

#ifdef USE_MPI
    std::cerr << "rank " << rank << " finished upward." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    logger::startTimer("Traverse");
    
#ifdef TAPAS_USE_VECTORMAP
    Tapas::Cell::TSPClass::Vectormap::vectormap_start();
#endif /*TAPAS_USE_VECTORMAP*/
    
    numM2L = 0; numP2P = 0;
    tapas::Map(FMM_DTT(), tapas::Product(*root, *root), args.mutual, args.nspawn, args.theta);
    
#ifdef TAPAS_USE_VECTORMAP
    vec3 Xperiodic = 0; // dummy; periodic not ported
    Tapas::Cell::TSPClass::Vectormap::vectormap_finish(P2P(),
                                                       *root,
                                                       Xperiodic);
#endif /*TAPAS_USE_VECTORMAP*/
    
    logger::stopTimer("Traverse");
    
    TAPAS_LOG_DEBUG() << "Dual Tree Traversal done\n";
    jbodies = bodies;

#ifdef TAPAS_DEBUG
    dumpL(*root);
#endif

    logger::startTimer("Downward pass");
    tapas::DownwardMap(FMM_Downward, *root);
    logger::stopTimer("Downward pass");
    
    TAPAS_LOG_DEBUG() << "L2P done\n";

#ifdef TAPAS_DEBUG
    dumpBodies(*root);
#endif
    
    CopyBackResult(bodies, root);
    //CopyBackResult(bodies, root->body_attrs(), args.numBodies);

    logger::printTitle("Total runtime");
    logger::stopPAPI();
    logger::stopTimer("Total FMM");
    logger::resetTimer("Total FMM");
    
#if WRITE_TIME
    logger::writeTime();
#endif

    const int numTargets = 100;

    logger::startTimer("Total Direct");
    CheckResult(bodies, numTargets, cycle, args.images);
    logger::stopTimer("Total Direct");
    
    buildTree.printTreeData(cells);
    logger::printPAPI();
    logger::stopDAG();
    bodies = bodies3;
    data.initTarget(bodies);

    root->Report();
  } /* end for */

#ifdef USE_MPI
  MPI_Finalize();
#endif
  
  return 0;
}


