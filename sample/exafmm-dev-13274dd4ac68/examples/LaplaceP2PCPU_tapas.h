#include "tapas/debug_util.h"
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

inline bool Close(double a, double b) { // for debug
  double a1 = a * 0.999;
  double a2 = a * 1.001;
  if (a1 < a2) return a1 <= b && b <= a2;
  else         return a2 <= b && b <= a1;
}

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
        
        {
          if (!getenv("TAPAS_IN_LET") && Bj.cell().key() == 2377900603251621891 && Close(Bj->X[0], -8.354853e-01)) {
            tapas::debug::DebugStream e("p2p");
            e.out() << Bj.cell().key() << " ";
            e.out() << Bi.cell().key() << " ";
            e.out() << "Bj: " << Bj->X << " ";
            e.out() << "Bi: " << Bi->X << " ";
            e.out() << "attr_j(1): ";
            for (int i = 0; i < 4; i++) {
              e.out() << std::setprecision(10) << std::fixed << Bj.attr()[i] << " ";
            }
            //e.out() << "dX: " << dX << " "; // ok
            //e.out() << "invR: " << invR << " "; //ok
            //e.out() << "pot: " << pot << " "; // ok
            e.out() << std::endl;
          }
        }
      }
    }
  }
};

#endif /*TAPAS_USE_VECTORMAP*/
