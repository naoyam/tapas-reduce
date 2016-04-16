#ifndef __TAPAS_HOT_GEOMETRY_H__
#define __TAPAS_HOT_GEOMETRY_H__

#include <iterator>
#include <type_traits>

#include "tapas/vec.h"
#include "tapas/basic_types.h"

namespace tapas {

template<typename VEC>
bool Separated(const VEC &xmax, const VEC &xmin, const VEC &ymax, const VEC &ymin) {
  const constexpr int Dim = VEC::Dim;

  bool separated = false;

  for (int d = 0; d < Dim; d++) {
    separated |= (xmax[d] <= ymin[d] || ymax[d] <= xmin[d]);
  }

  return separated;
}

template<typename Region>
bool Separated(const Region &x, const Region &y) {
  const constexpr int Dim = Region::Dim;

  bool separated = false;

  for (int d = 0; d < Dim; d++) {
    separated |= (x.max(d) <= y.min(d) || y.max(d) <= x.min(d));
  }

  return separated;
}

template<typename Iter, typename Region>
bool Separated(Iter beg, Iter end, Region &y) {
  using value_type = typename std::iterator_traits<Iter>::value_type;
  static_assert(std::is_same<typename std::remove_const<value_type>::type, typename std::remove_const<Region>::type>::value,
                "Inconsistent Types");

  bool r = true;
  for (Iter iter = beg; iter != end; iter++) {
    r = r && Separated(*iter, y);
  }

  return r;
}

// Returns if X includes Y
template<typename VEC>
bool Includes(VEC &xmax, VEC &xmin, VEC &ymax, VEC &ymin) {
  const constexpr int Dim = VEC::Dim;
  bool res = true;

  for (int d = 0; d < Dim; d++) {
    res &= (xmax[d] >= ymax[d] && ymin[d] >= xmin[d]);
  }

  return res;
}

class CenterClass {} Center;
class EdgeClass{} Edge;

template<int Dim, typename DIST_TYPE, typename FP> // takes DistanceType
struct Distance;

/**
 * \brief Struct to provide center-based distance functions
 */
template<int _DIM, typename _FP>
struct Distance<_DIM, CenterClass, _FP> {
  static const constexpr int Dim = _DIM;
  using FP = _FP;
  using Reg = Region<Dim, FP>;

  template<typename Cell>
  static inline FP Calc(Cell &c1, Cell &c2) {
    return (c1.center() - c2.center()).norm();
  }

  static inline FP CalcApprox(const Reg &bb, const Reg &src) {
    const Reg *beg = &bb, *end = beg + 1;
    return CalcApprox(beg, end, src);
  }

  /**
   * \brief Calculate distance between a target cell and source cell, where
   *        the target cell is a pseudo-cell (or region of the local process)
   * \param beg Begin iterator of Container<Reg>
   * \param beg End iterator of Container<Reg>
   * \param src Source region
   */
  template<typename Iter>
  static inline FP CalcApprox(Iter beg, Iter end, const Reg &src) {
    // Iter::value_type == Reg
    using value_type = typename std::remove_const<typename std::iterator_traits<Iter>::value_type>::type;
    static_assert(std::is_same<value_type, Reg>::value, "Inconsistent Types");

    Vec<Dim, FP> trg_ctr, src_ctr;
    double norm = std::numeric_limits<FP>::max(); // minimum norm between BB and src

    for (auto iter = beg; iter != end; iter++) {
      const Reg &trg = *iter;
      for (int d = 0; d < Dim; d++) {
        FP Rt = trg.width(d);
        FP Rs = src.width(d);
        FP sctr = src_ctr[d] = src.center(d);

        if (Rs >= Rt) {
          //trg_ctr[d] = trg.center(d);
          if (trg.center(d) >= src.center(d)) {
            trg_ctr[d] = std::max(trg.center(d) + Rt/2 - Rs/2, sctr);
          } else {
            trg_ctr[d] = std::min(trg.center(d) - Rt/2 + Rs/2, sctr);
          }
        } else if (sctr < trg.min(d) + Rs/2) {
          trg_ctr[d] = trg.min(d) + Rs/2;
        } else if (sctr > trg.max(d) - Rs/2) {
          trg_ctr[d] = trg.max(d) - Rs/2;
        } else {
          trg_ctr[d] = src_ctr[d];
        }
      }

      norm = std::min((src_ctr - trg_ctr).norm(), norm);
    }

    return norm;
  }
};

} // namespace tapas

#endif // __TAPAS_HOT_GEOMETRY_H__
