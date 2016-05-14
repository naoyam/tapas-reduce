#ifndef TAPAS_MAP_H_
#define TAPAS_MAP_H_

#include <cmath>
#include <vector>
#include <algorithm>
#include <tuple>
#include <functional>
#include <type_traits>

#include "tapas/debug_util.h"
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
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    dispatch(c1, c2, typename GenSeq<sizeof...(Args)>::type());
  }
  
  template<class CellType1, class CellType2, int...SS>
  INLINE void dispatch(CellType1 &c1, CellType2 &c2, Seq<SS...>) {
#ifdef TAPAS_COMPILER_INTEL
# pragma forceinline
#endif
    f_(c1, c2, std::get<SS>(args_)...);
  }
};

template<class T, class U, class V, class Iter1, class Iter2>
struct IfCell {
  static void debug(U&, V&) { }
};

template<class T, class Iter1, class Iter2>
struct IfCell<T,T,T,Iter1,Iter2> {
  static void debug(T &c1, T &c2) {
    if (getenv("TAPAS_IN_LET")) return;
    using CellType = T;
    using KeyType = typename CellType::SFC::KeyType;
    KeyType k1 = 2;
    KeyType k2 = 2594073385365405698;
    
    if ((c1.key() == k1 && c2.key() == k2) ||
        (c1.key() == k2 && c2.key() == k1)) {
      tapas::debug::DebugStream e("product_map");
      e.out() << "c1=" << c1.key() << "  c2=" << c2.key() << std::endl;
    }
  }
};

/**
 * Map function f over product of two iterators
 */
template <class Funct, class T1_Iter, class T2_Iter, class... Args>
void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
  using CellType = typename T1_Iter::CellType;
  if (prod.size() > 0) {
    CellType &c = *(prod.t1_);
    c.mapper().MapP2(f, prod, args...);
  }
}


template <class Funct, class T1_Iter, class...Args>
inline void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
  using CellType = typename T1_Iter::CellType;

  TAPAS_LOG_DEBUG() << "map product iterator size: "
                    << prod.size() << std::endl;
  
  if (prod.size() > 0) {
    CellType &c = prod.t1_.cell();
    c.mapper().MapP1(f, prod, args...);
  }
}

template <class Funct, class CellType, class... Args>
inline void Map(Funct f, tapas::iterator::SubCellIterator<CellType> iter, Args...args) {
  CellType &c = iter.cell();
  c.mapper().Map(f, iter, args...);
}

template <class Funct, class CellType, class... Args>
inline void Map(Funct f, iter::BodyIterator<CellType> iter, Args...args) {
  CellType &c = iter.cell();
  c.mapper().Map(f, iter, args...);
}

template <class Funct, class Iterator, class...Args>
inline void Map(Funct f, Iterator iter, Args...args) {
  auto &c = iter.cell();
  c.mapper().Map(f, iter, args...);
}

template<class Funct, class T, class...Args>
inline void UpwardMap(Funct f, T &x, Args...args) {
  // Deprecated:
  TAPAS_ASSERT(!"tapas::UpwardMap() is deprecated.");
  exit(-1);
  auto &c = x.cell();
  c.mapper().Map(f, x, args...);
}

template<class Funct, class T, class...Args>
inline void DownwardMap(Funct f, T &x, Args...args) {
  // Deprecated:
  TAPAS_ASSERT(!"tapas::DownwardMap() is deprecated.");
  exit(-1);
  auto &c = x.cell();
  c.mapper().DownwardMap(f, x, args...);
}

} // namespace tapas

#endif // TAPAS_MAP_H_
