#include <thread>
#include <mutex>
#include <unistd.h> // usleep

#ifdef EXAFMM_TAPAS_MPI
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

#define TAPAS

#ifdef TAPAS
#include "tapas_exafmm.h"
#ifdef TAPAS_USE_VECTORMAP
#include "LaplaceP2PCPU_tapas.cxx"
#endif /*TAPAS_USE_VECTORMAP*/
#include "serial_tapas_helper.cxx"
#endif

// Dump the M vectors of all cells.
void dumpM(Tapas::Cell &root) {
  std::stringstream ss;
#ifdef EXAFMM_TAPAS_MPI
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
#ifdef EXAFMM_TAPAS_MPI
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
#ifdef EXAFMM_TAPAS_MPI
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
#ifdef EXAFMM_TAPAS_MPI
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


#ifdef EXAFMM_TAPAS_MPI

template<class F>
void BarrierExec(F func) {
    int size, rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Barrier(MPI_COMM_WORLD);
    for (int i = 0; i < size; i++) {
        if (rank == i) {
            func();
        }
        usleep(10000);
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

#else

template<class F>
void BarrierExec(F func) {
    func();
}

#endif

int main(int argc, char ** argv) {
  Args args(argc, argv);
  
#ifdef EXAFMM_TAPAS_MPI
  int required = MPI_THREAD_MULTIPLE;
  int provided;

  MPI_Init_thread(&argc, &argv, required, &provided);

  if (provided < required) {
    std::cerr << "Your MPI implementation's support level of multi threading is insufficient. "
              << "We need  MPI_THREAD_MULTIPLE" << std::endl;
    MPI_Finalize();
    exit(-1);
  }
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

  Region tr;

  num_threads(args.threads);

  logger::startTimer("Dataset generation");
  bodies = data.initBodies(args.numBodies, args.distribution, args.mpi_rank, args.mpi_size);
  logger::stopTimer("Dataset generation");

  // Dump all bodies data for debugging
  {
    Stderr err("bodies");
    err.out() << "numBodies = " << args.numBodies << ", " << args.mpi_rank << ", " << args.mpi_size << std::endl;
    for (auto &b : bodies) {
      err.out() << b.X << " " << b.SRC << std::endl;
    }
  }
  
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
#ifdef TAPAS
    asn(tr, bounds);
    TAPAS_LOG_DEBUG() << "Bounding box: " << tr << std::endl;
#endif

    Tapas::Cell *root = Tapas::Partition(
        bodies.data(), bodies.size(), tr, args.ncrit);
#if 0
    {
      std::ofstream tapas_out("tapas_0.txt");
      for (int i = 0; i < args.numBodies; ++i) {
        tapas_out << root->body_attrs()[i] << std::endl;
      }
    }
#endif
    
#ifdef EXAFMM_TAPAS_MPI
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    logger::startTimer("Upward pass");
    tapas::Map(FMM_P2M, *root, args.theta);
    logger::stopTimer("Upward pass");
    TAPAS_LOG_DEBUG() << "P2M done\n";

#ifdef EXAFMM_TAPAS_MPI
    std::cerr << "rank " << rank << " finished upward." << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    dumpLeaves(*root);
    dumpM(*root);

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

    dumpBodies(*root);
    dumpL(*root);

    logger::startTimer("Downward pass");
    tapas::Map(FMM_L2P, *root);
    logger::stopTimer("Downward pass");
    
    TAPAS_LOG_DEBUG() << "L2P done\n";

    CopyBackResult(bodies, root);
    //CopyBackResult(bodies, root->body_attrs(), args.numBodies);
#if 0
    {
      std::ofstream tapas_out("tapas_final.txt");
      for (int i = 0; i < args.numBodies; ++i) {
        tapas_out << bodies[i].TRG << std::endl;
        //tapas_out << root->body_attrs()[i] << std::endl;
      }
    }
#endif
    
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
#endif
    
    std::cerr << "bodies2 = \n";
    for (auto &b : bodies2) {
      std::cerr << b.TRG[0] << " " << b.X[0] << " " << b.X[1] << " " << b.X[2] << "\n";
    }
    std::cerr << "\n";

#if 0
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
  }
    
#ifdef EXAFMM_TAPAS_MPI
  MPI_Finalize();
#endif
  
  return 0;
}
