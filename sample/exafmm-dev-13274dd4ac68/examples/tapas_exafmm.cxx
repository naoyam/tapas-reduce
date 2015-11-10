#include <thread>
#include <mutex>
#include <unistd.h> // usleep

#ifdef USE_MPI
# include <mpi.h>
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
#include "traversal.h"
#include "up_down_pass.h"
#include "verify.h"

#include "tapas_exafmm.h"

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
static inline void FMM_P2M(Tapas::Cell &c, real_t theta) {
  if (!c.IsLeaf()) {
    tapas::Map(FMM_P2M, c.subcells(), theta);
  }
  
  c.attr().R = 0;
  c.attr().M = 0;
  c.attr().L = 0;
  
#ifdef TAPAS_DEBUG
  {
    Stderr e("FMM_P2M");
    e.out() << Tapas::SFC::Simplify(c.key()) << " (1) " << c.IsLeaf() << " ";
    e.out() << "c.attr().R = " << std::fixed << std::setprecision(6) << c.attr().R << " ";
    e.out() << std::endl;
  }
#endif
  
  if (c.IsLeaf()) {
    tapas_kernel::P2M(c);
  } else {
    tapas_kernel::M2M(c);
  }
  
  for (int i = 0; i < 3; ++i) {
    c.attr().R = std::max(c.width(i), c.attr().R);
  }

  c.attr().R = c.attr().R / 2 * 1.00001; // see bounds2box func
  c.attr().R /= theta;
}

static inline void FMM_L2P(Tapas::Cell &c) {
  if (c.nb() == 0) return;
  if (!c.IsRoot()) tapas_kernel::L2L(c);
  if (c.IsLeaf()) {
    tapas::Map(tapas_kernel::L2P, c.bodies());
  } else {
    tapas::Map(FMM_L2P, c.subcells());
  }
}


static void FMM_M2L(Tapas::Cell &Ci, Tapas::Cell &Cj, int mutual, int nspawn);

static inline void tapas_splitCell(Tapas::Cell &Ci, Tapas::Cell &Cj, int mutual, int nspawn) {
  if (Cj.IsLeaf()) {
    assert(!Ci.IsLeaf());                                   //  Make sure Ci is not leaf
    //for (C_iter ci=Ci0+Ci->ICHILD; ci!=Ci0+Ci->ICHILD+Ci->NCHILD; ci++) {// Loop over Ci's children
    //traverse(ci, Cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
    //}                                                         //
    //End loop over Ci's children
    tapas::Map(FMM_M2L, tapas::Product(Ci.subcells(), Cj), mutual, nspawn);
  } else if (Ci.IsLeaf()) {                               // Else if Ci is leaf
    assert(!Cj.IsLeaf());                                   //  Make sure Cj is not leaf
    //for (C_iter cj=Cj0+Cj->ICHILD; cj!=Cj0+Cj->ICHILD+Cj->NCHILD; cj++) {// Loop over Cj's children
    //traverse(Ci, cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
    //}                                                         //
    //End loop over Cj's children
    tapas::Map(FMM_M2L, tapas::Product(Ci, Cj.subcells()), mutual, nspawn);
  } else if (Ci.nb() + Cj.nb() >= (tapas::index_t)nspawn || (mutual && Ci == Cj)) {// Else if cells are still large
    //TraverseRange traverseRange(this, Ci0+Ci->ICHILD, Ci0+Ci->ICHILD+Ci->NCHILD,// Instantiate recursive functor
    //Cj0+Cj->ICHILD, Cj0+Cj->ICHILD+Cj->NCHILD, Xperiodic, mutual, remote);
    //traverseRange();                                          //
    //Traverse for range of cell pairs
    tapas::Map(FMM_M2L, tapas::Product(Ci.subcells(), Cj.subcells()), mutual, nspawn);
  } else if (Ci.attr().R >= Cj.attr().R) {                                // Else if Ci is larger than Cj
    //for (C_iter ci=Ci0+Ci->ICHILD; ci!=Ci0+Ci->ICHILD+Ci->NCHILD; ci++) {// Loop over Ci's children
    //traverse(ci, Cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
    //}                                                         //  End loop over Ci's children
    tapas::Map(FMM_M2L, tapas::Product(Ci.subcells(), Cj), mutual, nspawn);
  } else {                                                    // Else if Cj is larger than Ci
    //for (C_iter cj=Cj0+Cj->ICHILD; cj!=Cj0+Cj->ICHILD+Cj->NCHILD; cj++) {// Loop over Cj's children
    // traverse(Ci, cj, Xperiodic, mutual, remote);            //   Traverse a single pair of cells
    //}                                                         //  End loop over Cj's children
    tapas::Map(FMM_M2L, tapas::Product(Ci, Cj.subcells()), mutual, nspawn);
  }
}

static inline void FMM_M2L(Tapas::Cell &Ci, Tapas::Cell &Cj, int mutual, int nspawn) {
  //static inline void FMM_M2L(TapasCell &Ci, TapasCell &Cj, int

  //mutual) {
  if (Ci.nb() == 0 || Cj.nb() == 0) return;
  vec3 dX;
  asn(dX, Ci.center() - Cj.center());
  real_t R2 = norm(dX);
  vec3 Xperiodic = 0; // dummy; periodic not ported

#ifdef TAPAS_DEBUG
  {
    Stderr e("FMM_M2L");
    real_t R = (Ci.attr().R+Cj.attr().R) * (Ci.attr().R+Cj.attr().R);
    e.out() << "Ci=" << Tapas::SFC::Simplify(Ci.key()) << "(" << Ci.nb() << ") "
            << "Cj=" << Tapas::SFC::Simplify(Cj.key()) << "(" << Cj.nb() << ") "
            << "Ci.attr().R=" << std::fixed << std::setprecision(8) << Ci.attr().R << " "
            << "Cj.attr().R=" << std::fixed << std::setprecision(8) << Cj.attr().R << " "
            << "R=" << R << " "
            << std::endl;
  }
#endif
  
  if (R2 > (Ci.attr().R+Cj.attr().R) * (Ci.attr().R+Cj.attr().R)) {                   // If distance is far enough
    //std::cerr << "M2L approx\n";
    numM2L++;
    tapas_kernel::M2L(Ci, Cj, Xperiodic, mutual);                   //  M2L kernel
  } else if (Ci.IsLeaf() && Cj.IsLeaf()) {            // Else if both cells are bodies
#ifdef TAPAS_USE_VECTORMAP
    tapas::Map(tapas_kernel::P2P(), tapas::Product(Ci.bodies(), Cj.bodies()), Xperiodic);
#else 
    tapas::Map(tapas_kernel::P2P, tapas::Product(Ci.bodies(), Cj.bodies()), Xperiodic);
#endif /*TAPAS_USE_VECTORMAP*/

    numP2P++;
  } else {                                                    // Else if cells are close but not bodies
    tapas_splitCell(Ci, Cj, mutual, nspawn);             //  Split cell and call function recursively for child
  }                                                           // End if for multipole acceptance
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


// Dump the M vectors of all cells.
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
    ofs << cell.IsLeaf() << " ";
    ofs << cell.attr().M << std::endl;
    mtx.unlock();
    tapas::Map(dump, cell.subcells());
  };
  tapas::Map(dump, root);
  ofs.close();
}

// Dump the L vectors of all cells.
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
    ofs << std::setw(20) << std::right << Tapas::SFC::Simplify(cell.key()) << " ";
    ofs << std::setw(3) << cell.depth() << " ";
    ofs << cell.IsLeaf() << " ";
    ofs << cell.attr().L << std::endl;
    mtx.unlock();
    tapas::Map(dump, cell.subcells());
  };
  tapas::Map(dump, root);
  ofs.close();
}

// Dump the body attrs of all cells
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
        ofs << iter->X << " ";
        ofs << iter->SRC << " ";
        for (int j = 0; j < 4; j++) {
          ofs << std::setw(20) << std::right << iter.attr()[j] << " ";
        }
        ofs << std::endl;
      }
      mtx.unlock();
    } else {
      tapas::Map(dump, cell.subcells());
    }
  };
  tapas::Map(dump, root);
  ofs.close();
}

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
  Traversal traversal(args.nspawn, args.images);
  
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
    Stderr err("bodies");
    err.out() << "numBodies = " << args.numBodies << ", " << args.mpi_rank << ", " << args.mpi_size << std::endl;
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
    
#ifdef USE_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    logger::startTimer("Upward pass");
    tapas::UpwardMap(FMM_P2M, *root, args.theta);
    logger::stopTimer("Upward pass");

#ifdef USE_MPI
    std::cerr << "rank " << rank << " finished upward." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
#endif

    //dumpLeaves(*root);
    //dumpM(*root);

    logger::startTimer("Traverse");
    
#ifdef TAPAS_USE_VECTORMAP
    Tapas::Cell::TSPClass::Vectormap::vectormap_start();
#endif /*TAPAS_USE_VECTORMAP*/
    
    numM2L = 0; numP2P = 0;
    tapas::Map(FMM_M2L, tapas::Product(*root, *root), args.mutual, args.nspawn);
    
#ifdef TAPAS_USE_VECTORMAP
    vec3 Xperiodic = 0; // dummy; periodic not ported
    Tapas::Cell::TSPClass::Vectormap::vectormap_finish(tapas_kernel::P2P(),
                                                       *root,
                                                       Xperiodic);
#endif /*TAPAS_USE_VECTORMAP*/
    
    logger::stopTimer("Traverse");
    
    TAPAS_LOG_DEBUG() << "M2L done\n";
    jbodies = bodies;

    //dumpBodies(*root);
    //dumpL(*root);

    logger::startTimer("Downward pass");
    tapas::Map(FMM_L2P, *root);
    logger::stopTimer("Downward pass");
    
    TAPAS_LOG_DEBUG() << "L2P done\n";

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
    bodies3 = bodies;
    data.sampleBodies(bodies, numTargets);
    bodies2 = bodies;
    data.initTarget(bodies);

#if 0
    std::cerr << "bodies (before direct)= \n";
    for (auto &b : bodies) {
      std::cerr << b.TRG[0] << " " << b.X[0] << " " << b.X[1] << " " << b.X[2] << "\n";
    }
    std::cerr << "\n";
    
    std::cerr << "bodies2 = \n";
    for (auto &b : bodies2) {
      std::cerr << b.TRG[0] << " " << b.X[0] << " " << b.X[1] << " " << b.X[2] << "\n";
    }
    std::cerr << "\n";

    std::cerr << "jbodies = \n";
    for (auto &b : jbodies) {
      std::cerr << b.TRG[0] << " " << b.X[0] << " " << b.X[1] << " " << b.X[2] << "\n";
    }
    std::cerr << "\n";
#endif
    
    logger::startTimer("Total Direct");
    traversal.direct(bodies, jbodies, cycle);
    traversal.normalize(bodies);
    logger::stopTimer("Total Direct");
    double potDif = verify.getDifScalar(bodies, bodies2);
    double potNrm = verify.getNrmScalar(bodies);
    double accDif = verify.getDifVector(bodies, bodies2);
    double accNrm = verify.getNrmVector(bodies);

#if 0
    std::cerr << "bodies (after direct)= ";
    for (auto &b : bodies) {
      std::cerr << b.TRG[0] << " " << b.X[0] << " " << b.X[1] << " " << b.X[2] << "\n";
    }
    std::cerr << "\n";
#endif
    
    std::cout << "potDif = " << potDif << std::endl;
    std::cout << "potNrm = " << potDif << std::endl;
    std::cout << "accDif = " << potDif << std::endl;
    std::cout << "accNrm = " << potDif << std::endl;
    
    logger::printTitle("FMM vs. direct");
    verify.print("Rel. L2 Error (pot)",std::sqrt(potDif/potNrm));
    verify.print("Rel. L2 Error (acc)",std::sqrt(accDif/accNrm));
    buildTree.printTreeData(cells);
    traversal.printTraversalData();
    logger::printPAPI();
    logger::stopDAG();
    bodies = bodies3;
    data.initTarget(bodies);
  } /* for */
    
#ifdef USE_MPI
  MPI_Finalize();
#endif
  
  return 0;
}

