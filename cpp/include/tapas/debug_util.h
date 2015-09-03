#ifndef TAPAS_DEBUG_UTIL_H_
#define TAPAS_DEBUG_UTIL_H_

#include <string>

#include "tapas/common.h"
#include "tapas/basic_types.h"

namespace tapas {
namespace debug {

template <int DIM, class FP, class BT>
void PrintBodies(const typename BT::type *b, int nb, std::ostream &os) {
  for (int i = 0; i < nb; ++i) {
    for (int j = 0; j < DIM; ++j) {    
      FP p = *ParticlePosOffset<DIM, FP, BT::pos_offset>::pos((void *)&(b[i]), j);
      os << p << " ";
    }
    os << std::endl;
  }
}

template <class T>
std::string ToStr(T v) {
  std::stringstream ss;
  ss << v;
  return ss.str();
}

} // debug
} // tapas

#endif // TAPAS_DEBUG_UTIL_H_
