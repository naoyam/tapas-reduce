#include "tapas_exafmm.h"

const real_t EPS2 = 0.0;                                        //!< Softening parameter (squared)

#ifdef TAPAS_USE_VECTORMAP

struct P2P {

  P2P() {}

#ifdef __CUDACC__
  __host__ __device__ __forceinline__
#endif
  void operator() (Body* Bi, Body* Bj, kvec4 &biattr, vec3 Xperiodic) {
    vec3 dX = Bi->X - Bj->X - Xperiodic;
    real_t R2 = norm(dX) + EPS2;
    if (R2 != 0) {
      real_t invR2 = 1.0 / R2;
      real_t invR = Bi->SRC * Bj->SRC * sqrt(invR2);
      dX *= invR2 * invR;
      biattr[0] += invR;
      biattr[1] -= dX[0];
      biattr[2] -= dX[1];
      biattr[3] -= dX[2];
    }
  }
};

#else /* TAPAS_USE_VECTORMAP */

struct P2P {
  template<class BodyIterator>
  void operator()(BodyIterator &Bi, BodyIterator &Bj, vec3 Xperiodic) {
    kreal_t pot = 0; 
    kreal_t ax = 0;
    kreal_t ay = 0;
    kreal_t az = 0;
    vec3 dX = Bi->X - Bj->X - Xperiodic;
    real_t R2 = norm(dX) + EPS2;

    if (R2 != 0) {
      auto attr = Bi.attr();
      real_t invR2 = 1.0 / R2;
      real_t invR = Bi->SRC * Bj->SRC * sqrt(invR2);
      dX *= invR2 * invR;
      pot += invR;
      ax += dX[0];
      ay += dX[1];
      az += dX[2];
      attr[0] += pot;
      attr[1] -= dX[0];
      attr[2] -= dX[1];
      attr[3] -= dX[2];
      Bi.attr() = attr; // Element-wise assignment to BodyAttribute is not allowed in Tapas
      if (Bi != Bj) {
        attr = Bj.attr();
        attr[0] += invR;
        attr[1] += dX[0];
        attr[2] += dX[1];
        attr[3] += dX[2];
        Bj.attr() = attr;
      }
    }
  }
};

#endif /*TAPAS_USE_VECTORMAP*/
