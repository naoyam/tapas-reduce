#ifndef TAPAS_HOT_MAPPER_H_
#define TAPAS_HOT_MAPPER_H_

#include <iostream>
#include <cxxabi.h>
#include <chrono>

#include "tapas/iterator.h"

extern "C" {
  // for performance debugging
  void myth_start_papi_counter(const char*);
  void myth_stop_papi_counter(void);
}


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

#if 0
  std::string T1_str, T2_str;
  {
    int status;
    char * t1_demangled = abi::__cxa_demangle(typeid(T1_Iter).name(),0,0,&status);
    char * t2_demangled = abi::__cxa_demangle(typeid(T2_Iter).name(),0,0,&status);
    if (strncmp("tapas::iterator::BodyIterator", t1_demangled, strlen("tapas::iterator::BodyIterator")) != 0 ||
        strncmp("tapas::iterator::BodyIterator", t2_demangled, strlen("tapas::iterator::BodyIterator")) != 0) {
      std::cout << "T1_Iter=" << (t1_demangled+17) << " "
                << "T2_Iter=" << (t2_demangled+17) << " "
                << "iter1.size()=" << iter1.size() << "[" << beg1 << "-" << end1 << "]" << "(th=" << T1_Iter::kThreadSpawnThreshold << ") "
                << "iter2.size()=" << iter2.size() << "[" << beg2 << "-" << end2 << "]" << "(th=" << T1_Iter::kThreadSpawnThreshold << ") "
                << ((end1 - beg1 <= kT1 && end2 - beg2 <= kT2) ? "Serial" : "Split")
                << std::endl;
    }
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

    typename Th::TaskGroup tg;
    tg.createTask([&]() { ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, end2, f, args...); });
    ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, end2, f, args...);
    tg.wait();
  } else if (end2 - beg2 == 1) {
    // opt_mutual == 1 && end2 - beg2 == 1
    int mid1 = (end1 + beg1) / 2;
    ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, end2, f, args...);
    ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, end2, f, args...);
  } else {
    int mid1 = (end1 + beg1) / 2;
    int mid2 = (end2 + beg2) / 2;
    // run (beg1,mid1) x (beg2,mid2) and (mid1,end1) x (mid2,end2) in parallel
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() { ProductMapImpl(mapper, iter1, beg1, mid1, iter2, beg2, mid2, f, args...); });
      ProductMapImpl(mapper, iter1, mid1, end1, iter2, mid2, end2, f, args...);
      tg.wait();
    }
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() {ProductMapImpl(mapper, iter1, beg1, mid1, iter2, mid2, end2, f, args...);});
      ProductMapImpl(mapper, iter1, mid1, end1, iter2, beg2, mid2, f, args...);
      tg.wait();
    }
  }
}

/**
 * \brief Overloaded version of ProductMapImpl for bodies x bodies.
 */
template<class CELL, class BODY, class LET, class Funct, class...Args>
static void ProductMapImpl(CPUMapper<CELL, BODY, LET> & /*mapper*/,
                           typename CELL::BodyIterator iter1,
                           int beg1, int end1,
                           typename CELL::BodyIterator iter2,
                           int beg2, int end2,
                           Funct f, Args... args) {
  TAPAS_ASSERT(beg1 < end1 && beg2 < end2);
  //using BodyIterator = typename CELL::BodyIterator;

  bool am = iter1.AllowMutualInteraction(iter2);

  CELL &c1 = iter1.cell();
  CELL &c2 = iter2.cell();
  //auto data = c1.data_ptr();
  auto *bodies1 = &c1.body(0);
  auto *bodies2 = &c2.body(0);
  auto *attrs1 = &c1.body_attr(0);
  auto *attrs2 = &c2.body_attr(0);
  //auto &bodies = &data->local_bodies_;
  //auto &attrs = data->local_body_attrs_;

  if (am) {
    for (int i = beg1; i < end1; i++) {
      for (int j = beg2; j <= i; j++) {
        if (1) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
          f(bodies1[i], attrs1[i], bodies2[j], attrs2[j], args...);
        }
      }
    }
  } else {
    for (int i = beg1; i < end1; i++) {
      for (int j = beg2; j < end2; j++) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
        f(bodies1[i], attrs1[i], bodies2[j], attrs2[j], args...);
      }
    }
  }
}

template<class Cell, class Body, class LET>
struct CPUMapper {
  bool opt_mutual_;

  enum class Map1Dir {
    None,
    Upward,
    Downward
  };

  Map1Dir map1_dir_;

  using KeyType = typename Cell::KeyType;
  using SFC = typename Cell::SFC;

  CPUMapper() : opt_mutual_(false), map1_dir_(Map1Dir::None) { }

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
   * CPUMapper::Map (SubcellIterator)
   */
  template <class Funct, class...Args>
  void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args...args) {
    using Th = typename Cell::Threading;
    typename Th::TaskGroup tg;

    TAPAS_ASSERT(iter.index() == 0);

    const auto &data = iter.cell().data();
    KeyType k = iter.cell().key();
    
    for (int i = 0; i < iter.size(); i++) {
      // In downward mode, if the subcells are out of the local process
      // (= the cell is a global leaf but not a local root)
      // just skip if.
      if (map1_dir_ == Map1Dir::Downward) {
        KeyType ck = SFC::Child(iter.cell().key(), i);
        if (data.ht_.count(ck) == 0) continue;
      }
      
      tg.createTask([=]() mutable { this->Map(f, *iter, args...); });
      iter++;
    }
    tg.wait();
  }

  // cell x cell
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, Cell &c2, Args... args) {
    SCOREP_USER_REGION_DEFINE(trav_handle)
    using Time = decltype(clock::now());
    Time net_bt, net_et;

    //c2.data().trav_used_src_key_.insert(c2.key());

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

    // myth_start_papi_counter("PAPI_FP_OPS");
    // Body of traverse
    f(c1, c2, args...);
    // myth_stop_papi_counter();

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
  template <class Funct, class...Args>
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

  // before running upward traversal from the root,
  // we need to run local upward first and communicate the global leaf values between processes.
  template<class Funct, class...Args>
  inline void StartUpwardMap(Funct f, Cell &c, Args...args) {
    auto &data = c.data();

    for (auto &k : data.lroots_) {
      // parallelizable?
      TAPAS_ASSERT(data.ht_.count(k) == 1);
      auto pcell = data.ht_[k];
      f(*pcell, args...);
    }
    
    Cell::ExchangeGlobalLeafAttrs(data.ht_gtree_, data.lroots_);
  }

  /**
   * CPUMapper::Map  (1-parameter)
   */
  template<class Funct, class...Args>
  inline void Map(Funct f, Cell &c, Args...args) {

    if (c.IsRoot()) {
      // Map() has just started.
      // Find the direction of the function f (upward or downward)
      if (map1_dir_ != Map1Dir::None) {
        std::cerr << "Tapas ERROR: Tapas' internal state seems to be corrupted. Map function is not thread-safe." << std::endl;
        abort();
      }
      
      auto dir = LET::FindMap1Direction(c, f, args...);
      
      switch(dir) {
        case LET::MAP1_UP:
#ifdef TAPAS_DEBUG
          if (c.data().mpi_rank_ == 0) std::cout << "In Upward: Determining 1-map direction (in upward) : UP => OK" << std::endl;
#endif
          map1_dir_ = Map1Dir::Upward;
          
          StartUpwardMap(f, c, args...); // Run local upward first
          f(c, args...);
          
          map1_dir_ = Map1Dir::None;
          return;
          
        case LET::MAP1_DOWN:
#ifdef TAPAS_DEBUG
          if (c.data().mpi_rank_ == 0) std::cout << "Downward is detected." << std::endl;
#endif
          
          map1_dir_ = Map1Dir::Downward;

          f(c, args...);
          
          map1_dir_ = Map1Dir::None;
          return;
          
        default:
          if (c.data().mpi_rank_ == 0) std::cout << "In Map-1: Determining 1-map direction (in upward) : NOT UP => Wrong !!!" << std::endl;
          TAPAS_ASSERT("Direction is Unknown.");
          return;
      }
    } else {
      auto &data = c.data();
      
      // Non-root cells
      if (map1_dir_ == Map1Dir::None) {
        std::cerr << "Tapas ERROR: Tapas' internal state seems to be corrupted. Map function is not thread-safe." << std::endl;
        abort();
      }

      if (map1_dir_ == Map1Dir::Upward) {
        // Upward
        // Traversal for local trees is already done in StartUpwardMap().
        // Here we just perform traversal in the global tree part.
        if (data.gleaves_.count(c.key()) == 0) { // Stop traversal if c is a global leaf.
          f(c, args...);
        }
      } else if (map1_dir_ == Map1Dir::Downward) {
        // non-local cells are eliminated in Map(SubcellIterator).
        f(c, args...);
      }
    }
  }
  

  /**
   * CPUMapper::DownwardMap (deperecated)
   */
  template<class Funct, class...Args>
  inline void DownwardMap(Funct f, Cell &c, Args...args) {
    // This function is deprecated.
    TAPAS_ASSERT(!"Deperecated function");
    abort();
    
    Cell::DownwardMap(f, c, args...);
  }

  inline void Setup() {  }

  inline void Start2() { }

  inline void Finish() {  }
}; // class CPUMapper


#ifdef __CUDACC__

#include "tapas/vectormap.h"
#include "tapas/vectormap_cuda.h"

template<class Cell, class Body, class LET>
struct GPUMapper : CPUMapper<Cell, Body, LET> {

  using Base = CPUMapper<Cell, Body, LET>;
  using Data = typename Cell::Data;
  using Vectormap = tapas::Vectormap_CUDA_Packed<Cell::Dim,
                                                 typename Cell::FP,
                                                 typename Cell::Body,
                                                 typename Cell::BodyAttr,
                                                 typename Cell::AttrType>;
  
  Vectormap vmap_;

  std::chrono::high_resolution_clock::time_point map2_all_beg_, map2_all_end_;

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

  template<class Funct, class...Args>
  inline void MapP2(Funct f, ProductIterator<BodyIterator<Cell>, BodyIterator<Cell>> prod, Args...args) {
    std::cout << "MapP2 (body)" << std::endl;
    if (prod.size() > 0) {
      vmap_.map2(f, prod, args...);
    }
  }
  
  template <class Funct, class T1_Iter, class ...Args>
  inline void MapP1(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
    if (prod.size() > 0) {
      //vmap_.map2(f, prod, args...);
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  template <class Funct, class ...Args>
  inline void MapP1(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args...args) {
    if (prod.size() > 0) {
      vmap_.map2(f, prod, args...);
    }
  }
  
  GPUMapper() : CPUMapper<Cell, Body, LET>() { }

  /**
   * \brief Specialization of Map() over body x body product for GPU
   */
  template <class Funct, class...Args>
  inline void Map(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args...args) {
    // Offload bodies x bodies interaction to GPU
    // not used?
    std::cout << "*** Map(body) correct one" << std::endl;
    vmap_.map2(f, prod, args...);
  }
  
  inline void Setup() {
    // Called in tapas/hot.h
    vmap_.Setup(64,31);
  }
  
  // GPUMapper::Start for 2-param Map()
  inline void Start2() {
    //std::cout << "*** Start2()" << std::endl;
    vmap_.Start2();
  }

  // GPUMapper::Finish for 2-param Map()
  inline void Finish() {
    //std::cout << "*** Finish()" << std::endl;
    vmap_.Finish2();
  }

  /**
   * @brief Initialization of 2-param Map()
   *
   * - Setup CUDA device and variables
   * - Construct & exchange LET
   */
  template <class Funct, class...Args>
  void Map2_Init(Funct f, Cell&c1, Cell&c2, Args...args) {
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
    }

    // -- initialize GPU for 2-Param Map()
#ifdef TAPAS_DEBUG
    std::cout << "Calling Start2()" << std::endl;
#endif
    Start2();

    // -- check
    if (this->Base::opt_mutual_) {
      if (data.mpi_rank_ == 0) {
        std::cerr << "[To Fix] Error: mutual is not supported in CUDA implementation" << std::endl;
      }
      exit(-1);
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
  void Map2_Finish(Funct, Cell &c1, Cell &c2, Args...) {
    auto &data = c1.data();
    Finish(); // Execute CUDA kernel

    // collect runtime information
    map2_all_end_  = std::chrono::high_resolution_clock::now();
    auto d = map2_all_end_ - map2_all_beg_;

    data.time_map2_dev = vmap_.time_device_call_;
    data.time_map2_all = std::chrono::duration_cast<std::chrono::microseconds>(d).count() * 1e-6;
  }

  /*
   * \brief Main routine of dual tree traversal (2-param Map())
   * GPUMapper::Map
   *
   * Cell x Cell
   */
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, Cell &c2, Args... args) {
    static std::chrono::high_resolution_clock::time_point t1, t2;

    //std::cout << "GPUMapper::Map(2)  " << c1.key() << ", " << c2.key() << std::endl;
    
    if (c1.IsRoot() && c2.IsRoot()) {
      Map2_Init(f, c1, c2, args...);
    }

    f(c1, c2, args...);

    if (c1.IsRoot() && c2.IsRoot()) {
      Map2_Finish(f, c1, c2, args...);
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
#if 0
    for (int i = 0; i < iter.size(); ++i) {
      f(*iter, iter.attr(), args...);
      iter++;
    }
#else
    vmap_.map1(f, iter, args...);
#endif
  }

  template<class Funct, class...Args>
  inline void Map(Funct f, Cell &c, Args...args) {
    Base::Map(f, c, args...);
  }
  
  /**
   * GPUMapper::Map (subcelliterator)
   */
  template <class Funct, class... Args>
  inline void Map(Funct f, tapas::iterator::SubCellIterator<Cell> iter, Args...args) {
    // nvcc cannot find this function in the Base (=CPUMapper) class, so it's explicitly written
    Base::Map(f, iter, args...);
  }

  template<class Funct, class...Args>
  inline void Map(Funct f, BodyIterator<Cell> b1, BodyIterator<Cell> b2, Args...args) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    std::cout << "*** Map(body) Wrong one!!" << std::endl;
    abort();
    f(*b1, b1.attr(), *b2, b2.attr(), args...);
  }
}; // class GPUMapper

#endif /* __CUDACC__ */

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_MAPPER_H_
