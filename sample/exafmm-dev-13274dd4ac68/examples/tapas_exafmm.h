#ifndef EXAFMM_TAPAS_COMMON_H_
#define EXAFMM_TAPAS_COMMON_H_

#include "types.h" // exafmm/include/types.h

#include "tapas.h"
#include "tapas/single_node_morton_hot.h" // Morton-key based single node partitioning
#include "tapas/morton_hot.h" // Morton-key based partitioning with MPI

struct CellAttr {
    real_t R;
    vecP M;
    vecP L;
};

typedef tapas::BodyInfo<Body, 0> BodyInfo;
#ifdef EXAFMM_TAPAS_MPI
#warning "Building MPI version"
typedef tapas::Tapas<3, real_t, BodyInfo, kvec4, CellAttr, tapas::MortonHOT> Tapas;
#else
#warning "Building single node version"
typedef tapas::Tapas<3, real_t, BodyInfo, kvec4, CellAttr, tapas::SingleNodeMortonHOT> Tapas;
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
