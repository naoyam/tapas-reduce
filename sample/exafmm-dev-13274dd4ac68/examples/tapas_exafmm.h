#ifndef EXAFMM_TAPAS_H_
#define EXAFMM_TAPAS_H_

#include "types.h" // exafmm/include/types.h
#include "tapas/debug_util.h"

#include "tapas.h"

struct CellAttr {
    real_t R;
    vecP M;
    vecP L;
};

template<class T, class U>
constexpr size_t MemberOffset(U T::*pmem) {
  return (char*)&((T*)nullptr->*pmem) - (char*)nullptr;
}

//
// Offset of coordinate values in Body
// ex.
// If body type is
//   struct MyBody {
//     vec3 X;
//   };
// then it's 0.
//
const constexpr size_t kBodyCoordOffset = MemberOffset(&Body::X);

// Select MPI or single-process
#ifdef USE_MPI
#include "tapas/hot.h"
#else
#include "tapas/single_node_hot.h"
#endif /* USE_MPI */

#ifdef MTHREADS
#warning "MTHREADS is defined. Do you mean \"MTHREAD\"?"
#endif

#ifndef MTHREAD
#warning "tapas_exafmm: building non-threaded code"
#endif

// Select threading component: serial or MassiveThreads
#ifdef MTHREAD
#include "tapas/threading/massivethreads.h"
using Threading = tapas::threading::MassiveThreads;
#endif /* MTHREAD */

#if 0 // ----------------------------------------------------------- Old TSP

typedef tapas::Tapas<3, real_t, Body, kBodyCoordOffset, kvec4, CellAttr,
                     HOT, Threading, Mapper
#ifdef TAPAS_USE_VECTORMAP
#  ifdef __CUDACC__
                     , tapas::Vectormap_CUDA_Packed<3, real_t, BodyInfo, kvec4>
#  else
                     , tapas::Vectormap_CPU<3, real_t, BodyInfo, kvec4>
#  endif /*__CUDACC__*/
#endif /*TAPAS_USE_VECTORMAP*/
                     > Tapas;

#else // ---------------------------------------------------------- New TSP

struct FMM_Params : public tapas::HOT<3, real_t, Body, kBodyCoordOffset, kvec4, CellAttr> {
#ifdef MTHREAD
  using Threading = tapas::threading::MassiveThreads;
#endif
};

using TapasFMM = tapas::Tapas2<FMM_Params>;

#endif // ------------------------------------------------------------

typedef TapasFMM::Region Region;

#if 0 // to be deleted 
namespace tapas_kernel {

void P2M(Tapas::Cell &C);
void M2M(Tapas::Cell &C);
void M2L(Tapas::Cell &Ci, Tapas::Cell &Cj, vec3 Xperiodic, bool mutual);
void L2L(Tapas::Cell &C);
void L2P(Tapas::BodyIterator &B);
#ifdef TAPAS_USE_VECTORMAP
struct P2P;
#else
void P2P(Tapas::BodyIterator &Ci, Tapas::BodyIterator &Cj, vec3 Xperiodic);
#endif /*TAPAS_USE_VECTORMAP*/

} // tapas_kernel
#endif /* if 0 */

#endif // EXAFMM_TAPAS_H_
