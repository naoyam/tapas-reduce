#ifndef __DEBUG_TAPAS_H__
#define __DEBUG_TAPAS_H__

#include "tapas_exafmm.h"

struct DumpM_Callback {
  std::ofstream &ofs_;
  std::mutex &mtx_;
  DumpM_Callback(std::ofstream &ofs, std::mutex &mtx) : ofs_(ofs), mtx_(mtx) { }
  
  template<class Cell>
  void operator()(Cell &cell) {
    mtx_.lock();
    ofs_ << std::setw(20) << std::right << TapasFMM::TSP::SFC::Simplify(cell.key()) << " ";
    ofs_ << std::setw(3) << cell.depth() << " ";
    ofs_ << (cell.IsLeaf() ? "L" : "_") << " ";
    ofs_ << cell.attr().M << std::endl;
    mtx_.unlock();
    TapasFMM::Map(*this, cell.subcells());
  }
};

// Debug function: Dump the M vectors of all cells.
void dumpM(TapasFMM::Cell &root) {
  std::stringstream ss;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << "M." << std::setw(4) << std::setfill('0') << rank << ".dat";
#else
  ss << "M.dat";
#endif
  
  std::ofstream ofs(ss.str().c_str());
  std::mutex mtx;
  TapasFMM::Map(DumpM_Callback(ofs, mtx), root);
  ofs.close();
}

struct DumpL_Callback {
  std::ofstream &ofs_;
  std::mutex &mtx_;

  DumpL_Callback(std::ofstream &ofs, std::mutex &mtx) : ofs_(ofs), mtx_(mtx) { }

  template<class Cell>
  void operator()(Cell &cell) {
    mtx_.lock();
    ofs_ << std::setw(20) << std::right << cell.key() << " ";
    ofs_ << std::setw(3) << std::noshowpos << cell.depth() << " ";
    ofs_ << (cell.IsLeaf() ? "L" : "_") << " ";

    for (size_t i = 0; i < cell.attr().L.size(); i++) {
      auto v = cell.attr().L[i];
      double real = v.real();
      double imag = v.imag();

      if (-1e-8 < real && real < 0) real = +0.0;
      if (-1e-8 < imag && imag < 0) imag = +0.0;

      ofs_ << "("
          << std::fixed << std::setprecision(5) << std::showpos << real << ","
          << std::fixed << std::setprecision(5) << std::showpos << imag
          << ") ";
    }
    //ofs_ << cell.attr().L << std::endl;
    ofs_ << std::endl;
    mtx_.unlock();

    TapasFMM::Map(*this, cell.subcells());
  }
};

// Debug function: Dump the L vectors of all cells.
void dumpL(TapasFMM::Cell &root) {
  std::stringstream ss;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << "L." << std::setw(4) << std::setfill('0') << rank << ".dat";
#else
  ss << "L.dat";
#endif
  
  std::ofstream ofs(ss.str().c_str());
  std::mutex mtx;
  TapasFMM::Map(DumpL_Callback(ofs, mtx), root);
  ofs.close();
}

struct DumpBodies_Callback {
  std::ofstream &ofs_;
  std::mutex &mtx_;

  DumpBodies_Callback(std::ofstream &ofs, std::mutex &mtx) : ofs_(ofs), mtx_(mtx) { }

  template<class Cell>
  void operator()(Cell &cell) {
    if (cell.IsLeaf()) {
      mtx_.lock();
      //ofs_ << std::setw(20) << std::right << TapasFMM::SFC::Simplify(cell.key()) << " ";
      auto iter = cell.bodies();
      for (int bi = 0; bi < (int)cell.nb(); bi++, iter++) {
        ofs_ << std::showpos << iter->X << " ";
        ofs_ << std::showpos << iter->SRC << " " << "vec4= ";
        for (int j = 0; j < 4; j++) {
          ofs_ << std::right
              << std::setiosflags(std::ios::showpos)
              << iter.attr()[j] << " ";
        }
        ofs_ << " " << cell.key() << std::endl;
      }
      mtx_.unlock();
    } else {
      TapasFMM::Map(*this, cell.subcells());
    }
  }
};

// Debug function: Dump the body attrs of all cells
void dumpBodies(TapasFMM::Cell &root) {
  std::stringstream ss;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << "bodies." << std::setw(4) << std::setfill('0') << rank << ".dat";
#else
  ss << "bodies.dat";
#endif
  
  std::ofstream ofs(ss.str().c_str());
  std::mutex mtx;
  TapasFMM::Map(DumpBodies_Callback(ofs, mtx), root);
  ofs.close();
}

struct DumpLeaves_Callback {
  std::ofstream &ofs_;
  std::mutex &mtx_;

  DumpLeaves_Callback(std::ofstream &ofs, std::mutex &mtx) : ofs_(ofs), mtx_(mtx) { }

  template<class Cell>
  void operator()(Cell &cell) {
    if (cell.IsLeaf()) {
      mtx_.lock();
      ofs_ << std::setw(20) << cell.key() << ", depth=" << cell.depth() << ", nb=" << cell.nb() << std::endl;
      for (int i = 0; i < (int)cell.nb(); i++) {
        ofs_ << "    body[" << i << "]=(" << cell.body(i).X << ") " << std::endl;
      }
      mtx_.unlock();
    } else {
      TapasFMM::Map(*this, cell.subcells());
    }
  }
};

// Debug function: dump all leaves
void dumpLeaves(TapasFMM::Cell &root) {
  std::stringstream ss;
#ifdef USE_MPI
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ss << "leaves." << std::setw(4) << std::setfill('0') << rank << ".dat";
#else
  ss << "leaves.dat";
#endif
  
  std::ofstream ofs(ss.str().c_str(), std::ios_base::app);
  std::mutex mtx;
  TapasFMM::Map(DumpLeaves_Callback(ofs, mtx), root);
  ofs.close();
}

#endif // __DEBUG_TAPAS_H__
