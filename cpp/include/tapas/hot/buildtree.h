/** \file
 *  Functions to construct HOT octrees
 */

#include "tapas/stdcbug.h"

#include <vector>

#include <mpi.h>

#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/mpi_util.h"

#ifndef TAPAS_HOT_BUILDTREE_H
#define TAPAS_HOT_BUILDTREE_H

namespace tapas {
namespace hot {

// Sampling rate
#ifndef TAPAS_SAMPLING_RATE
# define TAPAS_SAMPLING_RATE (1e-2)
#endif

template<class TSP, class SFC> struct HotData;
template<class TSP> class Cell;

/**
 * \class SamplingOctree
 * \brief Collection of static functions for sampling-based octree construction.
 * 
 */
template<class TSP, class SFC_>
struct SamplingOctree {
  static const constexpr int kDim = TSP::Dim;
  static const constexpr int kPosOffset = TSP::BT::pos_offset;
  
  using FP = typename TSP::FP;
  using SFC = SFC_;
  using KeyType = typename SFC::KeyType;
  using CellType = Cell<TSP>;
  using CellHashTable = typename std::unordered_map<KeyType, CellType*>;
  using KeySet = std::unordered_set<KeyType>;
  using BodyType = typename TSP::BT::type;
  using BodyAttrType = typename TSP::BT_ATTR;
  using Region = Region<TSP>;
  using Data = HotData<TSP, SFC>;

  static const constexpr double R = (TAPAS_SAMPLING_RATE);

  static Region ExchangeRegion(const Region &r) {
    typedef typename TSP::FP FP;
    
    Vec<kDim, FP> new_max, new_min;

    // Exchange max
    tapas::mpi::Allreduce(&r.max()[0], &new_max[0], kDim, MPI_MAX, MPI_COMM_WORLD);
    //::MPI_Allreduce(&r.max()[0], &new_max[0], kDim, MPI_DatatypeTraits<FP>::type(), MPI_MAX, MPI_COMM_WORLD);
  
    // Exchange min
    tapas::mpi::Allreduce(&r.min()[0], &new_min[0], kDim, MPI_MIN, MPI_COMM_WORLD);
    //::MPI_Allreduce(&r.min()[0], &new_min[0], kDim, MPI_DatatypeTraits<FP>::type(), MPI_MIN, MPI_COMM_WORLD);

    return Region(new_min, new_max);
  }


  /**
   * \brief Build an octree from bodies b, with a sampling-based method
   * \param[in] b     Unsorted bodies given from the user code.
   * \param[in] nb    Number of the bodies
   * \param[in] reg   A region object that are given to tapas::hot::Partition()
   * \param[in,out]   Data object
   */
  static void BuildTree(const BodyType *b, index_t nb, const Region& reg, std::shared_ptr<Data> data) {
    TAPAS_ASSERT(0.0 < R && R < 1.0);

    data->region_ = ExchangeRegion(reg);

    // sample particles
    auto sampled_keys = SampleKeys(b, nb, R, data->region_);

    // Gather the sampled particles into the DD-process
    int dd_proc_id = DDProcId();
  }

  /**
   * \brief Returns the rank of the domain decomposition process (DD-process)
   * For now, the rank 0 process always does this.
   */
  static int DDProcId() {
    return 0;
  }

  /**
   * \brief Sample (nb * R) bodies and returns their keys.
   * \param[in] b      A vector of bodies
   * \param[in] nb     Number of bodies
   * \param[in] R      Sampling ratio
   * \param[in] region Region of the global simulation space (returned by ExchangeRegion()).
   * \return           Sampled vector of keys 
   */
  static std::vector<KeyType> SampleKeys(const BodyType *b, index_t nb, double R, const Region &region) {
    size_t num_samples = nb * R;

    // sampled bodies
    std::vector<BodyType> bodies(b, b + num_samples);
    int num_finest_cells = 1 << SFC::MAX_DEPTH; // maximum number of cells in one dimension

    Vec<kDim, FP> pitch;           // possible minimum cell width
    for (int d = 0; d < kDim; ++d) {
      pitch[d] = (region.max()[d] - region.min()[d]) / (FP)num_finest_cells;
    }
    
    // convert the bodies into SFC keys
    std::vector<KeyType> keys(num_samples);

    for (size_t i = 0; i < num_samples; i++) {
      Vec<kDim, FP> ofst = ParticlePosOffset<kDim, FP, kPosOffset>::vec(reinterpret_cast<const void*>(b+i));
      ofst -= region.min(); // set the base 0
      ofst /= pitch;        // quantitize offsets

      Vec<kDim, int> anchor; // An SFC key-like, but SOA-format vector without depth information  (not that SFC keys are AOS format).
      
      // now ofst is a kDim-dimensional index of the finest-level cell to which the body belongs.
      for (int d = 0; d < kDim; d++) {
        anchor[d] = (int)ofst[d];

        if (anchor[d] == num_finest_cells) {
          // the body is just on the upper edge so anchor[d] is over the
          TAPAS_LOG_DEBUG() << "Particle located at max boundary." << std::endl;
          anchor[d]--;
        }
      }
      
      keys[i] = SFC::CalcFinestKey(anchor);
    }

    return keys;
  }
};

} // namespace hot 
} // namespace tapas


#endif // TAPAS_HOT_BUILDTREE_H

