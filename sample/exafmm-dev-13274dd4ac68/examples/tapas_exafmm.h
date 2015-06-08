#ifndef EXAFMM_TAPAS_COMMON_H_
#define EXAFMM_TAPAS_COMMON_H_

#include "types.h" // exafmm/include/types.h

#include "tapas.h"

#ifdef EXAFMM_TAPAS_MPI
#endif

struct CellAttr {
    real_t R;
    vecP M;
    vecP L;
};

typedef tapas::BodyInfo<Body, 0> BodyInfo;

#ifdef EXAFMM_TAPAS_MPI
// Build MPI-based distributed version
#include "tapas/morton_hot.h" // Morton-key based partitioning with MPI
#include "tapas/threading/massivethreads.h"
typedef tapas::Tapas<3, real_t, BodyInfo, kvec4, CellAttr,
                     tapas::HOT<3, tapas::key::Morton>,
                     tapas::threading::MassiveThreads> Tapas;
#else

// Build single-node version
#include "tapas/single_node_morton_hot.h" // Morton-key based single node partitioning
typedef tapas::Tapas<3, real_t, BodyInfo, kvec4, CellAttr,
                     tapas::SingleNodeHOT<3, tapas::key::Morton>,
                     tapas::threading::Serial> Tapas;

#endif

typedef Tapas::Region Region;

namespace tapas_kernel {

void P2M(Tapas::Cell &C);
void M2M(Tapas::Cell &C);
void M2L(Tapas::Cell &Ci, Tapas::Cell &Cj, vec3 Xperiodic, bool mutual);
void L2L(Tapas::Cell &C);
void L2P(Tapas::BodyIterator &B);
void P2P(Tapas::BodyIterator &Ci, Tapas::BodyIterator &Cj, vec3 Xperiodic);

} // tapas_kernel

#endif // EXAFMM_TAPAS_COMMON_H_
