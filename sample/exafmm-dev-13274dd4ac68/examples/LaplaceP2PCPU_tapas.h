#include "tapas/debug_util.h"
#include "tapas_exafmm.h"

const real_t EPS2 = 0.0;                                        //!< Softening parameter (squared)

extern uint64_t numP2P;

#ifdef COUNT
# define INC_P2P do { numP2P++; } while(0)
#else
# define INC_P2P
#endif

#ifdef USE_SCOREP
# include <scorep/SCOREP_User.h>
#else
#define SCOREP_USER_REGION(_1, _2)
#endif

#ifdef __CUDACC__

struct P2P {

  P2P() {}

  template<class _Body, class _BodyAttr>
#ifdef __CUDACC__
  __host__ __device__ __forceinline__
#endif
  void operator() (_Body& Bi, _BodyAttr &Bi_attr, _Body& Bj, _BodyAttr &Bj_attr, vec3 Xperiodic, int /*mutual*/) {
    vec3 dX = Bi.X - Bj.X - Xperiodic;
    real_t R2 = norm(dX) + EPS2;
    if (R2 != 0) {
      real_t invR2 = 1.0 / R2;
      real_t invR = Bi.SRC * Bj.SRC * sqrt(invR2);
      dX *= invR2 * invR;
      Bi_attr[0] += invR;
      Bi_attr[1] -= dX[0];
      Bi_attr[2] -= dX[1];
      Bi_attr[3] -= dX[2];
    }
  }
};

#else /* not __CUDACC__ */

struct P2P {
  template<class _Body, class _BodyAttr>
  void operator()(_Body &Bi, _BodyAttr &Bi_attr, _Body &Bj, _BodyAttr &Bj_attr, vec3 Xperiodic, int mutual) {
    INC_P2P;
    SCOREP_USER_REGION("P2P", SCOREP_USER_REGION_TYPE_FUNCTION);
    
    kreal_t pot = 0; 
    kreal_t ax = 0;
    kreal_t ay = 0;
    kreal_t az = 0;
    vec3 dX = Bi.X - Bj.X - Xperiodic;
    real_t R2 = norm(dX) + EPS2;

    if (R2 != 0) {
      auto attr = Bi_attr;
      real_t invR2 = 1.0 / R2;
      real_t invR = Bi.SRC * Bj.SRC * sqrt(invR2);
      dX *= invR2 * invR;
      pot += invR;
      ax += dX[0];
      ay += dX[1];
      az += dX[2];

      attr[0] += pot;
      attr[1] -= dX[0];
      attr[2] -= dX[1];
      attr[3] -= dX[2];
      Bi_attr = attr; // Element-wise assignment to BodyAttribute is not allowed in Tapas

      if (mutual && Bi.X != Bj.X) {
        attr = Bj_attr;
        attr[0] += invR;
        attr[1] += dX[0];
        attr[2] += dX[1];
        attr[3] += dX[2];
        Bj_attr = attr;
      }
    }
  }
};

#endif /* __CUDACC__ */
