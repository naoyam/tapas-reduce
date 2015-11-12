#ifndef TAPAS_HOT_DATA_H_
#define TAPAS_HOT_DATA_H_

namespace tapas {
namespace hot {

// fwd decl
template<class TSP> class Cell;
template<class TSP> class DummyCell;

/**
 * \brief Struct to hold shared data among Cells
 * Never accessed by users directly. Only held by Cells using shared_ptr.
 */
template<class TSP, class SFC_>
struct HotData {
  using SFC = SFC_;
  using KeyType = typename SFC::KeyType;
  using CellType = Cell<TSP>;
  using CellHashTable = typename std::unordered_map<KeyType, CellType*>;
  using KeySet = std::unordered_set<KeyType>;
  using BodyType = typename TSP::BT::type;
  using BodyAttrType = typename TSP::BT_ATTR;

#ifdef TAPAS_USE_VECTORMAP
  template <typename T>
  using vector_allocator = typename TSP::Vectormap:: template um_allocator<T>;
#endif /*TAPAS_USE_VECTORMAP*/

  CellHashTable ht_;
  CellHashTable ht_let_;
  CellHashTable ht_gtree_;  // Hsah table of the global tree.
  KeySet        gleaves_;   // set of global leaves, which are a part of ht_gtree_.keys and ht_.keys
  KeySet        lroots_;    // set of local roots. It must be a subset of gleaves. gleaves is "Allgatherv-ed" lroots.
  std::mutex ht_mtx_;  //!< mutex to protect ht_
  Region<TSP> region_; //!< global bouding box
  
  int mpi_rank_;
  int mpi_size_;
  int max_depth_; //!< Actual maximum depth of the tree

  std::vector<KeyType> leaf_keys_; //!< SFC keys of (all) leaves
  std::vector<index_t> leaf_nb_;   //!< Number of bodies in each leaf cell
  std::vector<int>     leaf_owners_; //!< Owner process of leaf[i]
#ifdef TAPAS_USE_VECTORMAP
  std::vector<BodyType, vector_allocator<BodyType>>
  local_bodies_; //!< Bodies that belong to the local process
#else /*TAPAS_USE_VECTORMAP*/
  std::vector<BodyType> local_bodies_; //!< Bodies that belong to the local process
#endif /*TAPAS_USE_VECTORMAP*/
  std::vector<KeyType>  local_body_keys_; //!< SFC keys of local bodies
#ifdef TAPAS_USE_VECTORMAP
  std::vector<BodyAttrType, vector_allocator<BodyAttrType>>
  local_body_attrs_; //!< Local body attributes
#else /*TAPAS_USE_VECTORMAP*/
  std::vector<BodyAttrType> local_body_attrs_; //!< Local body attributes
#endif /*TAPAS_USE_VECTORMAP*/

  std::vector<BodyType> let_bodies_;
  std::vector<BodyAttrType> let_body_attrs_;
  
  std::vector<KeyType> proc_first_keys_; //!< first SFC key of each process

  // log and time measurements (mainly of the local process)
  double sampling_rate; // sampling rate of tree construction
  index_t nb_total;  // total number of bodies.
  index_t nb_before; // local bodies before tree construction (given by the user)
  index_t nb_after;  // local bodies after tree construction  (actuall)
  index_t nleaves;   // number of leaves assigned to the local process
  index_t ncells;    // number of cells (note: some non-leaf cells are shared between processes)
  
  double time_tree_all;
  double time_tree_sample;     // Tree construction / sampling phase
  double time_tree_exchange;   // Tree construction / body exchange
  double time_tree_growlocal;  // Tree construction / grow local tree
  double time_tree_growglobal; // Tree construction / grow global tree

  double time_let_all;      // ExchangeLET/All
  double time_let_traverse; // ExchangeLET/Traverse
  double time_let_req;      // ExchangeLET/Request
  double time_let_response; // ExchangeLET/Response
  double time_let_register; // ExchangeLET/register
  
  HotData() { }
  HotData(const HotData<TSP, SFC>& rhs) = delete; // no copy
  HotData(HotData<TSP, SFC>&& rhs) = delete; // no move
};



} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_DATA_H_

