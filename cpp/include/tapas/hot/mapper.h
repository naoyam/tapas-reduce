#ifndef TAPAS_HOT_MAPPER_H_
#define TAPAS_HOT_MAPPER_H_

#include "tapas/iterator.h"
#include "tapas/hot/let.h"

namespace tapas {
namespace hot {


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
  
  if (end1 - beg1 <= kT1 || end2 - beg2 <= kT2) {
    // The two ranges (beg1,end1) and (beg2,end2) are fine enough to apply f in a serial manner.

    // Create a function object to be given to the Container's Map function.
    //typedef std::function<void(C1&, C2&)> Callback;
    //Callback gn = [&args...](C1 &c1, C2 &c2) { f()(c1, c2, args...); };
    for(int i = beg1; i < end1; i++) {
      for(int j = beg2; j < end2; j++) {
        T1_Iter lhs = iter1 + i;
        T2_Iter rhs = iter2 + j;
        // if i and j are mutually interactive, f(i,j) is evaluated only once.

        bool am = lhs.AllowMutualInteraction(rhs);
        
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

template<class Cell, class Body, class LET>
struct CPUMapper {
  using ProxyCell = typename LET::ProxyCell;
  using ProxyBody = typename LET::ProxyBody;
  using ProxyBodyIterator = typename LET::ProxyBodyIterator;
  
  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
    TAPAS_LOG_DEBUG() << "map product iterator size: "
                      << prod.size() << std::endl;
    
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  template <class Funct, class T1_Iter, class ...Args>
  inline void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
    TAPAS_LOG_DEBUG() << "map product iterator size: "
                      << prod.size() << std::endl;
    
    if (prod.size() > 0) {
      ProductMapImpl(*this,
                     prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }

#ifdef TAPAS_USE_VECTORMAP
  
  /* (Specialization of the Map() below by a general ProductIterator<T>
     with ProductIterator<BodyIterator<T>>). */
  template <class Funct, class...Args>
  void Map(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args...args) {
    typedef typename Cell::TSPClass::Vectormap Vectormap;
    Vectormap::vector_map2(f, prod, args...);
  }

#else
  
  // template <class Funct, class...Args>
  // void Map(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args...args) {
  // }
  
#endif
  
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
  
  // template <class Funct, class...Args>
  // void Map(Funct f, Cell &c1, Args...args) {
  //   f(c1, args...);
  // }

  // cell x cell
  template <class Funct, class...Args>
  void Map(Funct f, Cell &c1, Cell &c2, Args... args) {
    if (c1.IsRoot() && c2.IsRoot()) {
      std::cerr << "************************** Root x Root is called" << std::endl;
      if (c1.data().mpi_size_ > 1) {
        char t[] = "TAPAS_IN_LET=1";
        putenv(t); // to avoid warning "convertion from const char* to char*"
        LET::Exchange(c1, f, args...);
        unsetenv("TAPAS_IN_LET");
      }

#ifdef TAPAS_USE_VECTORMAP
      // -- CUDA initialization
      //Cell<TSP>::TSPClass::Vectormap::vectormap_start();
#endif
    }

    f(c1, c2, args...);
    
    if (c1.IsRoot() && c2.IsRoot()) {
#ifdef TAPAS_USE_VECTORMAP
      // vec3 Xperiodic = 0;
      // int mutual = c1.GetOptMutual();
      // assert(!mutual && "Mutual is not suppoted on CUDA");
      // Cell<TSP>::TSPClass::Vectormap::vectormap_finish(P2P(), c1, Xperiodic, mutual);
#endif
    }
  }

  // cell x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
    //f(c1, *c2, args...);
  }
  
  // cell iter x cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
    //f(*c1, *c2, args...);
  }

  // cell X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, Cell &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, c1, *c2, args...);
    //f(c1, *c2, args...);
  }

  // cell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, CellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
    //f(*c1, *c2, args...);
  }

  // subcell iter X cell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, CellIterator<Cell> &c2, Args...args) {
    Map(f, *c1, *c2, args...);
    //f(*c1, *c2, args...);
  }

  // subcell iter X subcell iter
  template <class Funct, class...Args>
  inline void Map(Funct f, SubCellIterator<Cell> &c1, SubCellIterator<Cell> &c2, Args...args) {
    //f(*c1, *c2, args...);
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

}; // class CPUMapper

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_MAPPER_H_

