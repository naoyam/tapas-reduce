#ifndef TAPAS_MAP_H_
#define TAPAS_MAP_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <functional>

#include "tapas/cell.h"
#include "tapas/iterator.h"

namespace {
namespace iter = tapas::iterator;
}

namespace tapas {

// Utility classes to wrap a user's function and arguments
// into a simple function object that takes only cells.
template<int ...SS>
struct Seq { };

template<int N, int...SS>
struct GenSeq : GenSeq<N-1, N-1, SS...> { };

template<int...SS>
struct GenSeq<0, SS...> {
  typedef Seq<SS...> type;
};

template<class Funct, class ...Args>
struct CallbackWrapper {
  Funct f_;
  std::tuple<Args...> args_;

  CallbackWrapper(Funct f, Args... args) : f_(f), args_(args...) {}

  template<class CellType1, class CellType2>
  void operator()(CellType1 &c1, CellType2 &c2) {
    dispatch(c1, c2, typename GenSeq<sizeof...(Args)>::type());
  }
  
  template<class CellType1, class CellType2, int...SS>
  void dispatch(CellType1 &c1, CellType2 &c2, Seq<SS...>) {
    f_(c1, c2, std::get<SS>(args_)...);
  }
};

#if 0
template<class Funct, class CellType1, class CellType2, class ...Args>
struct CallbackWrapper {
  Funct f_;
  std::tuple<Args...> args_;

  CallbackWrapper(Funct f, Args... args) : f_(f), args_(args...) {}

  void operator()(CellType1 &c1, CellType2 &c2) {
    dispatch(c1, c2, typename GenSeq<sizeof...(Args)>::type());
  }
  
  template<int...SS>
  void dispatch(CellType1 &c1, CellType2 &c2, Seq<SS...>) {
    f_(c1, c2, std::get<SS>(args_)...);
  }
};
#endif

/**
 * @brief A threshold to stop recursive tiling parallelization
 * Map for product uses checkerbord parallelism described in Taura et al.[1]
 * to avoid write conflicts and explicit mutual exclusions.
 * Split the product space until row or columns >= THRESHOLD.
 *
 * \todo Tune the value for each type (i.e. Body and Cells) or introduce auto-tuning
 *
 * [1] Taura et al. "A Task Parallelism Meets Fast Multipole Methods"
 */
template<class T>
struct ThreadSpawnThreshold {
    static const int Value = 1;
};

/**
 * @brief Determines if two containers are 'mutually interactive'
 * When calculating an interaction between two containers, sometimes we can save
 * computation by 'mutual interaction'.
 * This is the default implementation of a function to determine
 * if we can apply mutual interaction between the two containers.
 */
template<class C1, class C2>
struct AllowMutual {
  static bool value(C1, C2) {
    // Generally, two elements of different types are not mutual interactive.
    return false;
  }
};

/** 
 * @brief Specialization of AllowMutual for elements of a same container type
 */
template<class C1>
struct AllowMutual<C1, C1> {
  static bool value(C1 c1, C1 c2) {
    return c1.AllowMutualInteraction(c2);
  }
};

template<class T1_Iter, class T2_Iter, class Funct, class...Args>
static void product_map(T1_Iter iter1, int beg1, int end1,
                        T2_Iter iter2, int beg2, int end2,
                        Funct f, Args... args) {
  assert(beg1 < end1 && beg2 < end2);
  
  using CellType = typename T1_Iter::CellType;
  //using C1 = typename T1_Iter::value_type; // Container type (actually Body or Cell)
  //using C2 = typename T2_Iter::value_type;
  using Th = typename CellType::Threading;

  using Callback = CallbackWrapper<Funct, Args...>;
  Callback callback(f, args...);
  
  if (end1 - beg1 <= ThreadSpawnThreshold<T1_Iter>::Value ||
      end2 - beg2 <= ThreadSpawnThreshold<T2_Iter>::Value) {
    // The two ranges (beg1,end1) and (beg2,end2) are fine enough to apply f in a serial manner.

    // Create a function object to be given to the Container's Map function.
    //typedef std::function<void(C1&, C2&)> Callback;
    //Callback g = [&args...](C1 &c1, C2 &c2) { f()(c1, c2, args...); };

    for(int i = beg1; i < end1; i++) {
      for(int j = beg2; j < end2; j++) {
        // if i and j are mutually interactive, f(i,j) is evaluated only once.
        bool am = AllowMutual<T1_Iter, T2_Iter>::value(iter1, iter2);
        if ((am && i <= j) || !am) {
          if (iter1.IsLocal()) {
            CellType::template Map<Callback>(callback, *(iter1+i), *(iter2+j));
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
      tg.createTask([&]() { product_map(iter1, beg1, mid1, iter2, beg2, mid2, f, args...); });
      tg.createTask([&]() { product_map(iter1, mid1, end1, iter2, mid2, end2, f, args...); });
      tg.wait();
    }
    {
      typename Th::TaskGroup tg;
      tg.createTask([&]() {product_map(iter1, beg1, mid1, iter2, mid2, end2, f, args...);});
      tg.createTask([&]() {product_map(iter1, mid1, end1, iter2, beg2, mid2, f, args...);});
      tg.wait();
    }
  }
}

/**
 * Map function f over product of two iterators
 */
template <class Funct, class T1_Iter, class T2_Iter, class... Args>
void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
  TAPAS_LOG_DEBUG() << "map product iterator size: "
                    << prod.size() << std::endl;
      
  product_map(prod.t1_, 0, prod.t1_.size(),
              prod.t2_, 0, prod.t2_.size(),
              f, args...);
  
#if 0
  for (index_t i = 0; i < prod.size(); ++i) {
    f(prod.first(), prod.second(), args...);
    prod++;
  }
#endif
}
  
template <class Funct, class T1_Iter, class...Args>
void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
  TAPAS_LOG_DEBUG() << "map product iterator size: "
                    << prod.size() << std::endl;
#if 0
  // This implementation using product_map is currently disabled because
  // as of now 1-argument ProductIterator is not used.
  product_map(prod.t1_, 0, prod.t1_.size(),
              prod.t2_, 0, prod.t2_.size(),
              f, args...);
#else
  for (index_t i = 0; i < prod.size(); ++i) {
    f(prod.first(), prod.second(), args...);
    prod++;
  }
#endif
}

template <class Funct, class CellType, class... Args>
void Map(Funct f, tapas::iterator::SubCellIterator<CellType> iter, Args...args) {
  TAPAS_LOG_DEBUG() << "map non-product subcell iterator size: "
                    << iter.size() << std::endl;
  typedef typename CellType::Threading Th;
  
  // pack args... into a lambda closure
  std::function<void(CellType&)> lambda = [=](CellType &cell) { f(cell, args...); };
  
  typename Th::TaskGroup tg;
  for (int i = 0; i < iter.size(); i++) {
    tg.createTask([&]() { CellType::Map(lambda, *iter); });
    iter++;
  } 
  tg.wait();
}

template <class Funct, class CellType, class... Args>
void Map(Funct f, iter::BodyIterator<CellType> iter, Args...args) {
  TAPAS_LOG_DEBUG() << "map non-product body iterator size: "
                    << iter.size() << std::endl;
  for (int i = 0; i < iter.size(); ++i) {
    f(*iter, args...);
    iter++;
  }
}

template<class Funct, class T, class...Args>
void PostOrderMap(Funct f, T &x, Args...args) {
  std::function<void(T&)> lambda = [=](T& x) { f(x, args...); };
  T::PostOrderMap(x, lambda);
}

template<class Funct, class T, class...Args>
void UpwardMap(Funct f, T &x, Args...args) {
  PostOrderMap(f, x, args...);
  //std::function<void(T&)> lambda = [=](T& x) { f(x, args...); };
  //T::PostOrderMap(x, lambda);
}


template <class Funct, class T, class... Args>
void Map(Funct f, T &x, Args...args) {
  TAPAS_LOG_DEBUG() << "map non-iterator" << std::endl;

  std::function<void(T&)> lambda = [=](T& x) { f(x, args...); };
  T::Map(lambda, x);
}

} // namespace tapas

#endif // TAPAS_MAP_H_
