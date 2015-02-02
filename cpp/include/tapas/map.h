#ifndef TAPAS_MAP_H_
#define TAPAS_MAP_H_

#include <cmath>
#include <vector>
#include <algorithm>

#include "tapas/thread.h"
#include "tapas/cell.h"
#include "tapas/iterator.h"

namespace tapas {

namespace {

/**
 * @brief Remove redundunt elements in a std::vector. The vector must be sorted.
 * 
 * This way is much faster than using std::set.
 */
template<class T>
std::vector<T> uniq(const std::vector<T> &_v) {
  std::vector<T> v(_v);
  v.erase(unique(begin(v), end(v)), end(v));
  return v;
}

/**
 * @brief Compute remote peer process IDs in distributed product_map function
 * @param lha a vector of process IDs (including the local process) that own cell A.
 * @param rha a vector of process IDs that own cell B.
 * @param rank ID of the local process
 * @todo Select pairs that are "physically" close to each other.
 * 
 * When F(cell A, cell B) is called in product_map(), and A or B is not local,
 * we need to get information of the remote cell using isend/irecv from a remote process.
 * In our distributed HOT algorithm, multiple processes may have the remote cell information.
 * This function determines which remote peers to communicate.
 *
 * For example, cell A is owned by processes {0, 1, 2, 3} and
 * cell B is owned by processes {10, 11, 12, 13, 14, 15, 16},
 * then the matchings are:
 *   0 - 10, 11
 *   1 - 12, 13
 *   2 - 14, 15
 *   3 - 16
 * Process 0 communicates send cell A to process 10, 11 and receives cell B from 10, 11
 * using isend/irecv.
 */
std::vector<int> GetCellExchangePeer(const std::vector<int> &lhs,
                                     const std::vector<int> &rhs, int rank) {
  int i = std::find(std::begin(lhs), std::end(lhs), rank) - std::begin(lhs);
  assert(0 <= i && i < lhs.size());
  
  int a = lhs.size();
  int b = rhs.size();

  if (a == b) {
    return std::vector<int>(1, rhs[i]);
  } else if (a > b) {
    int q = a / b;
    int m = a % b;
    return std::vector<int>(1, rhs[i / (q + m)]);
  } else { // a < b
    int q = b / a;
    int m = b % a;
    int beg = q * i + std::min<int>(m, i);
    int num = q + (i < m ? 1 : 0);
    int end = beg + num;
    return std::vector<int>(std::begin(rhs) + beg,
                            std::begin(rhs) + end);
  }
}

} // anon namespace

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
    static bool value(C1 c1, C2 c2) {
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
    if (end1 - beg1 <= ThreadSpawnThreshold<T1_Iter>::Value ||
        end2 - beg2 <= ThreadSpawnThreshold<T2_Iter>::Value) {
        for(int i = beg1; i < end1; i++) {
            for(int j = beg2; j < end2; j++) {
                bool am = AllowMutual<T1_Iter, T2_Iter>::value(iter1, iter2);
                if ((am && i <= j) || !am) {
                    f(*(iter1+i), *(iter2+j), args...);
                }
            }
        }
    } else {
        int mid1 = (end1 + beg1) / 2;
        int mid2 = (end2 + beg2) / 2;
        // run (beg1,mid1) x (beg2,mid2) and (mid1,end1) x (mid2,end2) in parallel
        {
            mk_task_group;
            create_taskA(product_map(iter1, beg1, mid1, iter2, beg2, mid2, f, args...));
            create_taskA(product_map(iter1, mid1, end1, iter2, mid2, end2, f, args...));
            wait_tasks;
        }
        {
            mk_task_group;
            create_taskA(product_map(iter1, beg1, mid1, iter2, mid2, end2, f, args...));
            create_taskA(product_map(iter1, mid1, end1, iter2, beg2, mid2, f, args...));
            wait_tasks;
        }
    }
}

#define USE_NEW_PRODUCT_MAP

/**
 * Map function f over product of two iterators
 */
template <class Funct, class T1_Iter, class T2_Iter, class... Args>
void Map(Funct f, ProductIterator<T1_Iter, T2_Iter> prod, Args...args) {
  TAPAS_LOG_DEBUG() << "map product iterator size: "
                    << prod.size() << std::endl;
#ifdef USE_NEW_PRODUCT_MAP
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
  
template <class Funct, class T1_Iter, class...Args>
void Map(Funct f, ProductIterator<T1_Iter> prod, Args...args) {
  TAPAS_LOG_DEBUG() << "map product iterator size: "
                    << prod.size() << std::endl;
#ifdef USE_NEW_PRODUCT_MAP
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

template <class Funct, class T, class... Args>
void Map(Funct f, SubCellIterator<T> iter, Args...args) {
    TAPAS_LOG_DEBUG() << "map non-product subcell iterator size: "
                      << iter.size() << std::endl;
    mk_task_group;
    for (int i = 0; i < iter.size(); i++) {
        create_task0(f(*iter, args...));
        iter++;
    }
    wait_tasks;
}
  
template <class Funct, class T, class... Args>
void Map(Funct f, BodyIterator<T> iter, Args...args) {
  TAPAS_LOG_DEBUG() << "map non-product body iterator size: "
                    << iter.size() << std::endl;  
  for (int i = 0; i < iter.size(); ++i) {
    f(*iter, args...);
    iter++;
  }
}

template <class Funct, class T, class... Args>
void Map(Funct f, T &x, Args...args) {
  TAPAS_LOG_DEBUG() << "map non-iterator" << std::endl;
  f(x, args...);
}

} // namespace tapas

#endif // TAPAS_MAP_H_
