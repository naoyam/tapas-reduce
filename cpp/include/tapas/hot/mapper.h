#ifndef TAPAS_HOT_MAPPER_H_
#define TAPAS_HOT_MAPPER_H_

#include <iostream>
#include <cxxabi.h>
#include <chrono>

#include "tapas/iterator.h"
#include "tapas/hot/let.h"

namespace tapas {
namespace hot {

using std::chrono::duration_cast;
using std::chrono::milliseconds;
using clock = std::chrono::system_clock;

template<class Cell, class Body, class LET>struct CPUMapper;

/**
 * @brief Helper subroutine called from Mapper::Map
 */ 
template<class Mapper, class T1_Iter, class T2_Iter, class Funct, class...Args>
static void ProductMapImpl(Mapper &mapper,
                           T1_Iter iter1, int beg1, int end1,
                           T2_Iter iter2, int beg2, int end2,
                           Funct f, Args... args) {
  TAPAS_ASSERT(beg1 < end1 && beg2 < end2);
  
  using CellType = typename T1_Iter::CellType;
  using Th = typename CellType::Threading;

  const constexpr int kT1 = T1_Iter::kThreadSpawnThreshold;
  const constexpr int kT2 = T2_Iter::kThreadSpawnThreshold;

  bool am = iter1.AllowMutualInteraction(iter2);

  TAPAS_ASSERT(end1 > beg1 && end2 > beg2);

#if 0
  std::string T1_str, T2_str;
  {
    int status;
    char * t1_demangled = abi::__cxa_demangle(typeid(T1_Iter).name(),0,0,&status);
    char * t2_demangled = abi::__cxa_demangle(typeid(T2_Iter).name(),0,0,&status);
#if 0
    if (strncmp("tapas::iterator::BodyIterator", t1_demangled, strlen("tapas::iterator::BodyIterator")) != 0 ||
        strncmp("tapas::iterator::BodyIterator", t2_demangled, strlen("tapas::iterator::BodyIterator")) != 0) {
      std::cout << "T1_Iter=" << (t1_demangled+17) << " "
                << "T2_Iter=" << (t2_demangled+17) << " "
                << "iter1.size()=" << iter1.size() << "[" << beg1 << "-" << end1 << "]" << "(th=" << T1_Iter::kThreadSpawnThreshold << ") "
                << "iter2.size()=" << iter2.size() << "[" << beg2 << "-" << end2 << "]" << "(th=" << T1_Iter::kThreadSpawnThreshold << ") "
                << ((end1 - beg1 <= kT1 && end2 - beg2 <= kT2) ? "Serial" : "Split")
                << std::endl;
    }
#endif
    T1_str = t1_demangled;
    T2_str = t2_demangled;
    free(t1_demangled);
    free(t2_demangled);
  }
#endif
    
  if (!iter1.SpawnTask()
      || (end1 - beg1 == 1)
      || (end1 - beg1 <= kT1 && end2 - beg2 <= kT2)) {
    // Not to spawn tasks, run in serial
    // The two ranges (beg1,end1) and (beg2,end2) are fine enough to apply f in a serial manner.

    // Create a function object to be given to the Container's Map function.
    //typedef std::function<void(C1&, C2&)> Callback;
    //Callback gn = [&args...](C1 &c1, C2 &c2) { f()(c1, c2, args...); };
    for (int i = beg1; i < end1; i++) {
      for (int j = beg2; j < end2; j++) {
        T1_Iter lhs = iter1 + i;
        T2_Iter rhs = iter2 + j;
        // if i and j are mutually interactive, f(i,j) is evaluated only once.

        //bool am = lhs.AllowMutualInteraction(rhs);
        
        if ((am && i <= j) || !am) {
          if (lhs.IsLocal()) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
            mapper.Map(f, lhs, rhs, args...);
          }
        }
      }
    }
  } else if (!mapper.opt_mutual_ && end2 - beg2 == 1) {
    // Source side (iter2) can be split and paralleilzed.
    // target side cannot paralleize due to accumulation
    int mid1 = (end1 + beg1) / 2;

#if 0
    if (T1_str.find("BodyIterator") == std::string::npos && T2_str.find("BodyIterator") == std::string::npos) {
      std::cout << "T1_str=" << T1_str << ", "
                << "T2_str=" << T2_str << ", "
                << "beg1=" << beg1 << ", "
                << "end1=" << end1 << ", "
                << "mid1=" << mid1 << ", "
                << "mutual=" << mapper.opt_mutual_ << ", "
                << std::endl;
    }
#endif
        
    typename Th::TaskGroup tg;
    tg.createTask([&]() { ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, end2, f, args...); });
    tg.createTask([&]() { ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, end2, f, args...); });
    tg.wait();
  } else {
    int mid1 = (end1 + beg1) / 2;
    int mid2 = (end2 + beg2) / 2;
    // run (beg1,mid1) x (beg2,mid2) and (mid1,end1) x (mid2,end2) in parallel
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() { ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, mid2, f, args...); });
      tg.createTask([&]() { ProductMapImpl(mapper, iter1, mid1, end1, iter2, mid2, end2, f, args...); });
      tg.wait();
    }
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() {ProductMapImpl(mapper, iter1, beg1, mid1, iter2, mid2, end2, f, args...);});
      tg.createTask([&]() {ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, mid2, f, args...);});
      tg.wait();
    }
  }
}

#if 0
//
// FIXME: faster but wrong code: bodies are not in a single array, becuase there are bodies from remote processes
//

/**
 * \brief Overloaded version of ProductMapImpl for bodies x bodies.
 */
template<class CELL, class BODY, class LET, class Funct, class...Args>
static void ProductMapImpl(CPUMapper<CELL, BODY, LET> &mapper,
                           typename CELL::BodyIterator iter1,
                           int beg1, int end1,
                           typename CELL::BodyIterator iter2,
                           int beg2, int end2,
                           Funct f, Args... args) {
  TAPAS_ASSERT(beg1 < end1 && beg2 < end2);
  using BodyIterator = typename CELL::BodyIterator;

  bool am = iter1.AllowMutualInteraction(iter2);

  CELL &c1 = iter1.cell();
  CELL &c2 = iter2.cell();
  auto data = c1.data_ptr();
  auto &bodies = data->local_bodies_;
  auto &attrs = data->local_body_attrs_;

  if (am) {
    for (int i = beg1; i < end1; i++) {
      //BodyIterator b1 = iter1 + i;
      for (int j = beg2; j <= i; j++) {
        //BodyIterator b2 = iter2 + j;
        if (1) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
          //mapper.Map(f, b1, b2, args...);
          //f(*b1, b1.attr(), *b2, b2.attr(), args...);
          f(bodies[c1.bid() + i], attrs[c1.bid() + i],
            bodies[c2.bid() + j], attrs[c2.bid() + j], args...);
        }
      }
    }
  } else {
    for (int i = beg1; i < end1; i++) {
      //BodyIterator b1 = iter1 + i;
      for (int j = beg2; j < end2; j++) {
        //BodyIterator b2 = iter2 + j;
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
        //mapper.Map(f, b1, b2, args...);
        //f(*b1, b1.attr(), *b2, b2.attr(), args...);
        f(bodies[c1.bid() + i], attrs[c1.bid() + i],
          bodies[c2.bid() + j], attrs[c2.bid() + j], args...);
      }
    }
  }
}
#endif

template<class Cell, class Body, class LET>
struct CPUMapper {
  bool opt_mutual_;

  CPUMapper() : opt_mutual_(false) { }
  
  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void MapP2(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  template <class Funct, class T1_Iter, class ...Args>
  inline void MapP1(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }

  /**
   *
   */
  template <class Funct, class... Args>
  void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args...args) {
    using Th = typename Cell::Threading;
    typename Th::TaskGroup tg;

    TAPAS_ASSERT(iter.index() == 0);
    
    for (int i = 0; i < iter.size(); i++) {
      tg.createTask([=]() mutable { this->Map(f, *iter, args...); });
      iter++;
    } 
    tg.wait();
  }

  // cell x cell
  template <class Funct, class...Args>
  void Map(Funct f, Cell &c1, Cell &c2, Args... args) {
    SCOREP_USER_REGION_DEFINE(trav_handle)
    using Time = decltype(clock::now());
    Time net_bt, net_et;
    
    if (c1.IsRoot() && c2.IsRoot()) {
      // Pre-traverse procedure
      // All Map() function traverse starts from (root, root)
      // for LET construction and GPU init/finalize
      auto bt = clock::now();
      
      if (c1.data().mpi_size_ > 1) {
#ifdef TAPAS_DEBUG
        char t[] = "TAPAS_IN_LET=1";
        putenv(t); // to avoid warning "convertion from const char* to char*"
#endif
        LET::Exchange(c1, f, args...);
#ifdef TAPAS_DEBUG
        unsetenv("TAPAS_IN_LET");
#endif
      }

      SCOREP_USER_REGION_BEGIN(trav_handle, "NetTraverse", SCOREP_USER_REGION_TYPE_COMMON);
      
      auto et = clock::now();
      c1.data().time_map2_let = duration_cast<milliseconds>(et - bt).count() * 1e-3;
      
      net_bt = clock::now();
    }

    // Actual Map() operation
    f(c1, c2, args...);
    
    if (c1.IsRoot() && c2.IsRoot()) {
      // Post-traverse procedure
      net_et = clock::now();
      c1.data().time_map2_net = duration_cast<milliseconds>(net_et - net_bt).count() * 1e-3;
      SCOREP_USER_REGION_END(trav_handle);
    }
  }

  // cell x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
  }
  
  // cell iter x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // cell X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // bodies
  template <class Funct, class... Args>
  void Map(Funct f, BodyIterator<Cell> iter, Args...args) {
    for (int i = 0; i < iter.size(); ++i) {
      f(*iter, iter.attr(), args...);
      iter++;
    }
  }
  
  // body x body 
  template<class Funct, class...Args>
  void Map(Funct f, BodyIterator<Cell> b1, BodyIterator<Cell> b2, Args...args) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    f(*b1, b1.attr(), *b2, b2.attr(), args...);
  }

  inline void Setup() {  }
  
  inline void Start() {  }

  inline void Finish() {  }
}; // class CPUMapper


#ifdef __CUDACC__

#include "tapas/vectormap.h"
#include "tapas/vectormap_cuda.h"

template<class Cell, class Body, class LET>
struct GPUMapper {

  using Data = typename Cell::Data;
  using Vectormap = tapas::Vectormap_CUDA_Packed<Cell::Dim, typename Cell::FP, typename Cell::Body, typename Cell::BodyAttr>;

  Vectormap vmap_;

  std::chrono::high_resolution_clock::time_point map2_all_beg_, map2_all_end_;

  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  template <class Funct, class T1_Iter, class ...Args>
  inline void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  /* (Specialization of the Map() below by a general ProductIterator<T>
     with ProductIterator<BodyIterator<T>>). */
  template <class Funct, class...Args>
  inline void Map(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args...args) {
    vmap_.map2(f, prod, args...);
  }

  inline void Setup() {
    vmap_.setup(64,31);
  }

  // GPUMapper::Start for 2-param Map()
  inline void Start() {
    vmap_.start();
  }

  // GPUMapper::Finish for 2-param Map()
  inline void Finish() {
    vmap_.finish();
  }

  /**
   *
   */
  template <class Funct, class... Args>
  inline void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args...args) {
    using Th = typename Cell::Threading;
    typename Th::TaskGroup tg;

    TAPAS_ASSERT(iter.index() == 0);
    
    for (int i = 0; i < iter.size(); i++) {
      tg.createTask([=]() mutable { this->Map(f, *iter, args...); });
      iter++;
    } 
    tg.wait();
  }

  /**
   * @brief Initialization of 2-param Map()
   *
   * - Setup CUDA device and variables
   * - Construct & exchange LET
   */ 
  template <class Funct, class...Args>
  void Map2Init(Funct f, Cell&c1, Cell&c2, Args...args) {
    auto &data = c1.data();
    map2_all_beg_ = std::chrono::high_resolution_clock::now();

    // -- Perform LET exchange if more than 1 MPI process
    if (c1.data().mpi_size_ > 1) {
#ifdef TAPAS_DEBUG
      char t[] = "TAPAS_IN_LET=1";
      putenv(t); // to avoid warning "convertion from const char* to char*"
#endif
      LET::Exchange(c1, f, args...);
        
#ifdef TAPAS_DEBUG
      unsetenv("TAPAS_IN_LET");
#endif
    } else {
      // mpi_size_ == 0
      data.time_let_all = data.time_let_traverse
                        = data.time_let_req
                        = data.time_let_response
                        = data.time_let_register
                        = 0;
    }

    // -- initialize GPU
    Start();

    // -- check
    if (c1.GetOptMutual()) {
      std::cerr << "[To Fix] Error: mutual is not supported in CUDA implementation" << std::endl;
      //exit(-1);
    }
    data.time_map2_let = data.time_let_all;
  }

  /**
   * @brief Finalization of 2-param Map()
   *
   * - Execute CUDA kernel on the interaction list
   * - Collect time information
   */ 
  template <class Funct, class...Args>
  void Map2Finish(Funct, Cell &c1, Cell &c2, Args...) {
    auto &data = c1.data();
    Finish(); // Execute CUDA kernel

    // collect runtime information
    map2_all_end_  = std::chrono::high_resolution_clock::now();
    auto d = map2_all_end_ - map2_all_beg_;
    
    data.time_map2_dev = vmap_.time_device_call_;
    data.time_map2_all = std::chrono::duration_cast<std::chrono::microseconds>(d).count() * 1e-6;
  }
  
  // GPUMapper::Map
  // cell x cell
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, Cell &c2, Args... args) {
    static std::chrono::high_resolution_clock::time_point t1, t2;
    
    if (c1.IsRoot() && c2.IsRoot()) {
      Map2Init(f, c1, c2, args...);
    }
    
    f(c1, c2, args...);
    
    if (c1.IsRoot() && c2.IsRoot()) {
      Map2Finish(f, c1, c2, args...);
    }
  }

  // cell x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
  }
  
  // cell iter x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // cell X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
  }

  // bodies
  template <class Funct, class... Args>
  inline void Map(Funct f, BodyIterator<Cell> iter, Args...args) {
    for (int i = 0; i < iter.size(); ++i) {
      f(*iter, iter.attr(), args...);
      iter++;
    }
  }
  
  // body x body 
  template<class Funct, class...Args>
  inline void Map(Funct f, BodyIterator<Cell> b1, BodyIterator<Cell> b2, Args...args) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    f(*b1, b1.attr(), *b2, b2.attr(), args...);
  }

}; // class GPUMapper

#endif /* __CUDACC__ */

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_MAPPER_H_

