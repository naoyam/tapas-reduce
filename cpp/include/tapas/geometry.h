#ifndef __TAPAS_HOT_GEOMETRY_H__
#define __TAPAS_HOT_GEOMETRY_H__

#include "tapas/vec.h"

namespace tapas {

template<typename VEC>
bool Separated(VEC &xmax, VEC &xmin, VEC &ymax, VEC &ymin) {
  const constexpr int Dim = VEC::Dim;
  
  bool separated = false;

  for (int d = 0; d < Dim; d++) {
    separated |= (xmax[d] <= ymin[d] || ymax[d] <= xmin[d]);
  }

  return separated;
}

class CenterClass {} Center;
class EdgeClass{} Edge;

template<typename DIST_TYPE, typename FP> // takes DistanceType
struct Distance;

/**
 * \brief Struct to provide center-based distance functions
 */
template<typename FP>
struct Distance<CenterClass, FP> {
  template<typename Cell>
  static inline FP Calc(Cell &c1, Cell &c2) {
    return (c1.center() - c2.center()).norm();
  }

  /**
   * \brief Calculate distance between a target cell and source cell, where
   *        the target cell is a pseudo-cell (or region of the local process)
   */
  template<int DIM>
  static inline FP CalcApprox(Vec<DIM, FP> &trg_max,
                              Vec<DIM, FP> &trg_min,
                              Vec<DIM, FP> &src_max,
                              Vec<DIM, FP> &src_min) {
    Vec<DIM, FP> trg_ctr, src_ctr;    
    for (int d = 0; d < DIM; d++) {
      FP Rt = trg_max[d] - trg_min[d];
      FP Rs = src_max[d] - src_min[d];
      FP sctr = src_ctr[d] = (src_max[d] + src_min[d]) / 2;
      
      if (Rs >= Rt) {
        trg_ctr[d] = (trg_max[d] + trg_min[d]) / 2;
      } else if (sctr < trg_min[d] + Rs/2) {
        trg_ctr[d] = trg_min[d] + Rs/2;
      } else if (sctr > trg_max[d] - Rs/2) {
        trg_ctr[d] = trg_max[d] - Rs/2;
      } else {
        trg_ctr[d] = sctr;
      }
    }
    
    return (src_ctr - trg_ctr).norm();
  }
};

} // namespace tapas

#endif // __TAPAS_HOT_GEOMETRY_H__
