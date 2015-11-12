/** \file
 *  Functions to construct HOT octrees by a sampling method
 */

#ifndef TAPAS_HOT_BUILDTREE_H
#define TAPAS_HOT_BUILDTREE_H

#include "tapas/stdcbug.h"

#include <vector>
#include <iterator>
#include <algorithm>

#include <mpi.h>

#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/mpi_util.h"
#include "tapas/hot/global_tree.h"

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
class SamplingOctree {
 public:
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
  using Data = HotData<TSP, SFC>;
  template<class T> using Region = tapas::Region<T>;

 private:
  std::vector<BodyType> bodies_;
  std::vector<KeyType> body_keys_;
  std::vector<KeyType> proc_first_keys_; // first key of each process's region
  Region<TSP> region_;
  std::shared_ptr<Data> data_;
  int ncrit_;
            
 public:
  SamplingOctree(const BodyType *b, index_t nb, const Region<TSP>& reg, std::shared_ptr<Data> data, int ncrit)
      : bodies_(b, b+nb), body_keys_(), proc_first_keys_(), region_(reg), data_(data), ncrit_(ncrit)
  {
  }
    
  void ExchangeRegion() {
    Vec<kDim, FP> new_max, new_min;

    // Exchange max
    tapas::mpi::Allreduce(&region_.max()[0], &new_max[0], kDim, MPI_MAX, MPI_COMM_WORLD);
  
    // Exchange min
    tapas::mpi::Allreduce(&region_.min()[0], &new_min[0], kDim, MPI_MIN, MPI_COMM_WORLD);

    region_ = Region<TSP>(new_min, new_max);
  }
  

  /**
   * \brief Split the finest-level keys into `mpi_size` groups. Used in the DD-process.
   *
   * \param keys Keys of sampled bodies
   * \param mpi_size Number of processes
   */
  static std::vector<KeyType> PartitionSpace(const std::vector<KeyType> &keys, int mpi_size) {
    // Split L-level keys into `mpi_size` groups.
    //
    // We determine L by
    //     B = 2^Dim  (B = 8 in 3 dim space)
    //     Np = mpi_size
    //     L = log_B(Np) + 2
    //
    // L must be larger than the number of processes, and large enough to achieve good load balancing.
    // However, too large L leads to unnecessary deep tree structure because domain boundary may be too 
    // close to a certain particle.

    const int B = 1 << kDim;
    const int L  = log(mpi_size) / log(B) + 2; // logB(Np) = log(Np) / log(B)
    const KeyType K = SFC::AppendDepth(0, L);
    const int W = pow(B, L); // number of cells in level L

    const int q = keys.size() / mpi_size; // each process should have roughly q bodies
    
    std::vector<int> nb(W); // number of bodies each L-level key owns

    KeyType kl = K;

    // For each key(cell) in level L, count how many bodies belong to the cell out of the sampled set.
    for (size_t i = 0; i < nb.size(); i++) {
      index_t b, e;
      SFC::FindRangeByKey(keys, kl, b, e);
      nb[i] = e - b;
      kl = SFC::GetNext(kl);
    }

    // sum(nb) must be equal to keys.size(), becuase the for loop above cover the entire domain
    TAPAS_ASSERT(std::accumulate(nb.begin(), nb.end(), 0) == keys.size());
        
    std::vector<int> nb_iscan(W); // inclusive scan of nb vector

    for (size_t i = 0; i < nb_iscan.size(); i++) {
      // Future work: this operation can be parallelized by parallel scan
      nb_iscan[i] = nb[i] + (i > 0 ? nb_iscan[i-1] : 0);
    }
    
    std::vector<KeyType> beg_keys(mpi_size); // return value
    
    // find domain boundaries:
    // process i's beginning key is j-th key in level L, where j is the smallest index satisfying
    //   nb_iscan[j] >= q * i
    beg_keys[0] = SFC::AppendDepth(0, L);
    for (size_t i = 1; i < mpi_size; i++) {
      int j = std::upper_bound(nb_iscan.begin(), nb_iscan.end(), q*i) - nb_iscan.begin();
      beg_keys[i] = SFC::GetNext(K, j);
    }

    return beg_keys;
  }

  static double SamplingRate() {
    double R = 0.01;
    
#ifdef TAPAS_SAMPLING_RATE
    R = (TAPAS_SAMPLING_RATE);
#endif
    
    if (getenv("TAPAS_SAMPLING_RATE")) {
      R = atof(getenv("TAPAS_SAMPLING_RATE"));
    }
    
    TAPAS_ASSERT(0.0 < R && R < 1.0);

    return R;
  }

  /**
   * \brief Sample bodies and determine proc_first_keys_. 
   * Output: proc_first_keys_.
   */
  void Sample() {
    const double R = SamplingRate();
    double beg = MPI_Wtime();

    // todo:
    // record R

    ExchangeRegion();
    data_->region_ = region_;

    // sample particles in this process
    int sample_nb = bodies_.size() * R;
    std::vector<BodyType> sampled_bodies = std::vector<BodyType>(bodies_.begin(), bodies_.begin() + sample_nb);
    std::vector<KeyType> sampled_keys_local = BodiesToKeys(sampled_bodies, data_->region_);
    std::vector<KeyType> sampled_keys;
    
    // Gather the sampled particles into the DD-process
    int dd_proc_id = DDProcId();

    tapas::mpi::Gather(sampled_keys_local, sampled_keys, dd_proc_id, MPI_COMM_WORLD);

    std::sort(std::begin(sampled_keys), std::end(sampled_keys));

    proc_first_keys_.resize(data_->mpi_size_);

    if (data_->mpi_rank_ == dd_proc_id) {
      // in DD-process
      TAPAS_ASSERT(SFC::GetDepth(sampled_keys[0]) == SFC::MAX_DEPTH);
      
      proc_first_keys_ = PartitionSpace(sampled_keys, data_->mpi_size_);
      TAPAS_ASSERT(proc_first_keys_.size() == data_->mpi_size_);
    }

    // Each process's starting key is broadcast.
    tapas::mpi::Bcast(proc_first_keys_, dd_proc_id, MPI_COMM_WORLD);
    TAPAS_ASSERT(proc_first_keys_.size() == data_->mpi_size_);

    double end = MPI_Wtime();
    data_->time_tree_sample = end - beg;
  }

  /** 
   * \brief Exchange bodies to owner processes determined by Sample() function.
   */ 
  void Exchange() {
    double beg = MPI_Wtime();
    
    // Exchange bodies according to proc_first_keys_
    // new_bodies is the received bodies
    bodies_ = ExchangeBodies(bodies_, proc_first_keys_, region_, MPI_COMM_WORLD);
    body_keys_ = BodiesToKeys(bodies_, region_);

    // Sort both new_keys and new_bodies.
    SortByKeys(body_keys_, bodies_);

    // todo: record bodies

    data_->local_bodies_ = bodies_;
    data_->local_body_keys_ = body_keys_;
    data_->local_body_attrs_.resize(bodies_.size());
    bzero(data_->local_body_attrs_.data(), sizeof(BodyAttrType) * bodies_.size());

    double end = MPI_Wtime();
    data_->time_tree_exchange = end - beg;
  }
  
  /**
   * \brief Build an octree from bodies b, with a sampling-based method
   */
  void Build() {
    double beg = MPI_Wtime();
    
    Sample();
      
    Exchange();

    GrowLocal();

    // Get the max depth
    int d = data_->max_depth_;
    tapas::mpi::Allreduce(&d, &data_->max_depth_, 1, MPI_MAX, MPI_COMM_WORLD);
    data_->proc_first_keys_ = std::move(proc_first_keys_);

    // error check
    if (data_->ht_[0] == nullptr) {
      // If no leaf is assigned to the process, root node is not generated
      if (data_->mpi_rank_ == 0) {
        std::cerr << "There are too few particles compared to the number of processes."
                  << std::endl;
      }
      MPI_Finalize();
      exit(-1);
    }
    double end = MPI_Wtime();
    data_->time_tree_all = end - beg;
  }

  void GrowLocal() {
    double beg = MPI_Wtime();
    
    proc_first_keys_.push_back(SFC::GetNext(0));
    
    GenerateCell((KeyType)0, std::begin(body_keys_), std::end(body_keys_));
    
    proc_first_keys_.pop_back();

    double end = MPI_Wtime();
    data_->time_tree_growlocal = end - beg;
  }

  void GenerateCell(KeyType k,
                    typename std::vector<KeyType>::const_iterator pbeg, // beg of a subset of bkeys
                    typename std::vector<KeyType>::const_iterator pend // end of a subset of bkeys
                    ) {
    KeyType k2 = SFC::GetNext(k);
    auto bbeg = body_keys_.begin();
    
    int rank = data_->mpi_rank_;
      
    // range of bodies that belong to the cell k.
    auto range_beg = std::lower_bound(pbeg, pend, k);
    auto range_end = std::lower_bound(pbeg, pend, k2);

    int nb = range_end - range_beg;
    // Checks if the cell is included in the process or strides over two processes.
    // If the cell strides, it's never a leaf and must be split, even if nb <= ncrit.
    bool included = SFC::Includes(proc_first_keys_[rank], proc_first_keys_[rank+1], k);

    // if (nb <= ncrit) and this process owns the cell (i.e. no other process owns any descendants of the cell),
    // the cell is a leaf.
    bool is_leaf = (nb <= ncrit_) && included;
    int body_beg = is_leaf ? range_beg - bbeg : 0;

    // Construct a cell.
    CellType *c = new CellType(data_->region_, body_beg, nb);
    c->key_ = k;
    c->is_leaf_ = is_leaf;
    c->is_local_ = true;
    c->is_local_subtree_ = false;
    c->data_ = data_;
    c->bid_ = body_beg;
    c->nb_ = nb;
    bzero(&c->attr_, sizeof(c->attr_));

    TAPAS_ASSERT(nb >= 0);
    TAPAS_ASSERT(body_beg >= 0);

    data_->ht_[k] = c;

    if (SFC::GetDepth(k) > data_->max_depth_) {
      data_->max_depth_ = SFC::GetDepth(k);
    }

    if (is_leaf) {
      // The cell [k] is a leaf.
      data_->leaf_keys_.push_back(k);
      data_->leaf_nb_.push_back(nb);
      // todo remove Data::leaf_owners_
    } else {
      // The cell [k] is not a leaf. Split it again.
      // Note: if the cell is not a leaf and nb == 0, that means other processes may have particles which belong to the cell.
      auto ch_keys = SFC::GetChildren(k);
      
      for (auto chk : ch_keys) {
        // Check if the child key is in the range of this process, ignore it otherwise.
        bool overlap = SFC::Overlapped(chk, SFC::GetNext(chk),
                                       proc_first_keys_[rank],
                                       proc_first_keys_[rank+1]);
        
        // Note: SFC::GetNext(0) is the next key of the root key 0, which means
        //       the `end` of the whole region
        
        if (overlap) {
          GenerateCell(chk, range_beg, range_end);
        }
      }
    }
  }
  
  static std::vector<BodyType> ExchangeBodies(std::vector<BodyType>& bodies,
                                              const std::vector<KeyType> proc_first_keys,
                                              const Region<TSP> &reg, MPI_Comm comm) {
    std::vector<KeyType> body_keys = BodiesToKeys(bodies, reg);
    std::vector<int> dest(body_keys.size()); // destiantion of each body

    auto b = proc_first_keys.begin();
    auto e = proc_first_keys.end();

    for (size_t i = 0; i < body_keys.size(); i++) {
      dest[i] = std::upper_bound(b, e, body_keys[i]) - b - 1; // destination process
      TAPAS_ASSERT(0 <= dest[i]);
    }

    tapas::SortByKeys(dest, bodies);

    // exchange bodies using Alltoallv
    std::vector<BodyType> recv_bodies;
    std::vector<int> src;

    tapas::mpi::Alltoallv(bodies, dest, recv_bodies, src, comm);

    return recv_bodies;
  }
  
  /**
   * \brief Returns the rank of the domain decomposition process (DD-process)
   * For now, the rank 0 process always does this.
   */
  static int DDProcId() {
    return 0;
  }

  static std::vector<KeyType> BodiesToKeys(const std::vector<BodyType> &bodies, const Region<TSP> &region) {
    return BodiesToKeys(bodies.begin(), bodies.end(), region);
  }
  
  /**
   * \brief Transform a vector of bodies into a vector of Kyes
   * \param[in] bodies A vector of bodies
   * \param[in] region Region of the global simulation space (returned by ExchangeRegion()).
   * \return           Vector of keys
   */
  template<class Iter>
  static std::vector<KeyType> BodiesToKeys(Iter beg, Iter end, const Region<TSP> &region) {
    int num_finest_cells = 1 << SFC::MAX_DEPTH; // maximum number of cells in one dimension

    std::vector<KeyType> keys; // return value

    Vec<kDim, FP> pitch;           // possible minimum cell width
    for (int d = 0; d < kDim; ++d) {
      pitch[d] = (region.max()[d] - region.min()[d]) / (FP)num_finest_cells;
    }

    auto ins = std::back_inserter(keys);
    
    for (auto iter = beg; iter != end; iter++) {
      Vec<kDim, FP> ofst = ParticlePosOffset<kDim, FP, kPosOffset>::vec(reinterpret_cast<const void*>(&*iter));
      ofst -= region.min();
      ofst /= pitch;

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

      *ins = SFC::CalcFinestKey(anchor);
    }

    return keys;
  }
};

} // namespace hot 
} // namespace tapas


#endif // TAPAS_HOT_BUILDTREE_H

