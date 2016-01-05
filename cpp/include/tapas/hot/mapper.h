#ifndef TAPAS_HOT_MAPPER_H_
#define TAPAS_HOT_MAPPER_H_

#include "tapas/iterator.h"
#include "tapas/hot/let.h"

namespace tapas {
namespace hot {


/**
 * @brief Helper subroutine called from Mapper::Map
 */ 
template<class T1_Iter, class T2_Iter, class Funct, class...Args>
static void ProductMapImpl(T1_Iter iter1, int beg1, int end1,
                           T2_Iter iter2, int beg2, int end2,
                           Funct f, Args... args) {
  assert(beg1 < end1 && beg2 < end2);
  auto &mapper = (*iter1).cell().mapper();
  
  using CellType = typename T1_Iter::CellType;
  //using C1 = typename T1_Iter::value_type; // Container type (actually Body or Cell)
  //using C2 = typename T2_Iter::value_type;
  using Th = typename CellType::Threading;

  using Callback = CallbackWrapper<Funct, Args...>;
  Callback callback(f, args...);

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
            //f(*lhs, *rhs, args...);
            mapper.Map(callback, *lhs, *rhs);
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
      tg.createTask([&]() { ProductMapImpl(iter1, beg1, mid1, iter2, beg2, mid2, f, args...); });
      tg.createTask([&]() { ProductMapImpl(iter1, mid1, end1, iter2, mid2, end2, f, args...); });
      tg.wait();
    }
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() {ProductMapImpl(iter1, beg1, mid1, iter2, mid2, end2, f, args...);});
      tg.createTask([&]() {ProductMapImpl(iter1, mid1, end1, iter2, beg2, mid2, f, args...);});
      tg.wait();
    }
  }
}

template<class LET>
struct CPUMapper {
  /**
   * @brief Map function f over product of two iterators
   */
  template <class Funct, class T1_Iter, class T2_Iter, class... Args>
  inline void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
    TAPAS_LOG_DEBUG() << "map product iterator size: "
                      << prod.size() << std::endl;
    
    if (prod.size() > 0) {
      ProductMapImpl(prod.t1_, 0, prod.t1_.size(),
                     prod.t2_, 0, prod.t2_.size(),
                     f, args...);
    }
  }
  
  template <class Funct, class T1_Iter, class ...Args>
  inline void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
    TAPAS_LOG_DEBUG() << "map product iterator size: "
                      << prod.size() << std::endl;
    
    if (prod.size() > 0) {
      ProductMapImpl(prod.t1_, 0, prod.t1_.size(),
                  prod.t2_, 0, prod.t2_.size(),
                  f, args...);
    }
  }

#ifdef TAPAS_USE_VECTORMAP
  
  /* (Specialization of the Map() below by a general ProductIterator<T>
     with ProductIterator<BodyIterator<T>>). */
  template <class Funct, class Cell, class...Args>
  void Map(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args...args) {
    typedef typename Cell::TSPClass::Vectormap Vectormap;
    Vectormap::vector_map2(f, prod, args...);
  }

#else
  
  // template <class Funct, class Cell, class...Args>
  // void Map(Funct f, ProductIterator<BodyIterator<Cell>> prod, Args...args) {
  // }
  
#endif
  
  /**
   *
   */
  template <class Funct, class CellType, class... Args>
  void Map(Funct f, tapas::iterator::SubCellIterator<CellType> iter, Args...args) {
    TAPAS_LOG_DEBUG() << "map non-product subcell iterator size: "
                      << iter.size() << std::endl;
    typedef typename CellType::Threading Th;
  
    // pack args... into a lambda closure
    // FIXME: replace lambda with auto
    std::function<void(CellType&)> lambda = [=](CellType &cell) { f(cell, args...); };
  
    typename Th::TaskGroup tg;
    for (int i = 0; i < iter.size(); i++) {
      tg.createTask([&]() { CellType::Map(lambda, *iter); });
      iter++;
    } 
    tg.wait();
  }

  template <class Funct, class CellType>
  void Map(Funct f, CellType &c1) {
    f(c1);
  }

  template<class Funct, class CellType, class...Args>
  void Map(Funct f, BodyIterator<CellType> b1, BodyIterator<CellType> b2) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    f(b1, b2);
  }

  template <class Funct, class CellType, class...Args>
  void Map(Funct f, CellType &c1, CellType &c2) {
    if (c1.IsRoot() && c2.IsRoot()) {
      if (c1.data().mpi_size_ > 1) {
        char t[] = "TAPAS_IN_LET=1";
        putenv(t); // to avoid warning "convertion from const char* to char*"
        LET::Exchange(f, c1);
        unsetenv("TAPAS_IN_LET");
      }

#ifdef TAPAS_USE_VECTORMAP
      // -- CUDA initialization
      //Cell<TSP>::TSPClass::Vectormap::vectormap_start();
#endif
    }

    f(c1, c2);
    if (c1.IsRoot() && c2.IsRoot()) {
#ifdef TAPAS_USE_VECTORMAP
      // vec3 Xperiodic = 0;
      // int mutual = c1.GetOptMutual();
      // assert(!mutual && "Mutual is not suppoted on CUDA");
      // Cell<TSP>::TSPClass::Vectormap::vectormap_finish(P2P(), c1, Xperiodic, mutual);
#endif
    }
  }
  
  /**
   * 
   */
  template <class Funct, class CellType, class... Args>
  void Map(Funct f, BodyIterator<CellType> iter, Args...args) {
    TAPAS_LOG_DEBUG() << "map non-product body iterator size: "
                      << iter.size() << std::endl;
    for (int i = 0; i < iter.size(); ++i) {
      f(*iter, args...);
      iter++;
    }
  }
};

} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_MAPPER_H_

