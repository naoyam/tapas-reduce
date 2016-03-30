#ifndef TAPAS_HOT_DATA_H_
#define TAPAS_HOT_DATA_H_

namespace tapas {
namespace hot {

// fwd decl
template<class TSP> class Cell;
template<class TSP> class DummyCell;

/**
 * \brief Struct to hold shared data among Cells
 */
template<class TSP, class SFC_>
struct SharedData {
  using SFC = SFC_;
  using FP = typename TSP::FP;
  using KeyType = typename SFC::KeyType;
  using CellType = Cell<TSP>;
  using CellHashTable = typename std::unordered_map<KeyType, CellType*>;
  using KeySet = std::unordered_set<KeyType>;
  using BodyType = typename TSP::Body;
  using BodyAttrType = typename TSP::BodyAttr;
  using Mapper = typename CellType::Mapper;
  static const constexpr int Dim = SFC::Dim;

  template<class T> using Allocator = typename TSP::template Allocator<T>;

  CellHashTable ht_;
  CellHashTable ht_let_;
  CellHashTable ht_gtree_;  // Hsah table of the global tree.
  KeySet        gleaves_;   // set of global leaves, which are a part of ht_gtree_.keys and ht_.keys
  KeySet        lroots_;    // set of local roots. It must be a subset of gleaves_. gleaves_ is "Allgatherv-ed" lroots.
  std::mutex ht_mtx_;  //!< mutex to protect ht_
  Region<TSP> region_; //!< global bouding box
  Mapper mapper_;
  
  int mpi_rank_;
  int mpi_size_;
  MPI_Comm mpi_comm_;
  int max_depth_; //!< Actual maximum depth of the tree

  Vec<Dim, FP> local_bb_max_; //!< Coordinates of the bounding box of the local process
  Vec<Dim, FP> local_bb_min_; //!< Coordinates of the bounding box of the local process
  
  std::vector<KeyType> leaf_keys_; //!< SFC keys of (all) leaves
  std::vector<index_t> leaf_nb_;   //!< Number of bodies in each leaf cell
  std::vector<int>     leaf_owners_; //!< Owner process of leaf[i]
  
  std::vector<BodyType, Allocator<BodyType>> local_bodies_; //!< Bodies that belong to the local process
  std::vector<BodyType, Allocator<BodyType>> let_bodies_; //!< Bodies sent from remote processes
  std::vector<BodyAttrType, Allocator<BodyAttrType>> local_body_attrs_; //!< Local body attributes
  std::vector<BodyAttrType, Allocator<BodyAttrType>> let_body_attrs_; //!< Local body attributes
  
  std::vector<KeyType>  local_body_keys_; //!< SFC keys of local bodies
  
  std::vector<KeyType> proc_first_keys_; //!< first SFC key of each process

  int opt_task_spawn_threshold_;
  bool opt_mutual_;

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
  double time_let_req_all;      // ExchangeLET/Request
  double time_let_req_comm;     // ExchangeLET/Request
  double time_let_res_all;
  double time_let_res_attr_comm;
  double time_let_res_body_comm;
  double time_let_res_comp1; // ExchangeLET/Response
  double time_let_res_comm; // ExchangeLET/Response
  double time_let_res_comp2; // ExchangeLET/Response
  double time_let_register; // ExchangeLET/register

  double time_map2_all;
  double time_map2_let;     // Map2/LET (should be equivalent to time_let_all)
  double time_map2_net;
#ifdef __CUDACC__
  double time_map2_dev;  // CUDA kernel runtime
#endif

  std::unordered_map<int, int> let_func_count;

  SharedData()
      : mpi_rank_(0)
      , mpi_size_(1)
      , mpi_comm_(MPI_COMM_WORLD)
      , max_depth_(0)
      , opt_task_spawn_threshold_(1000)
      , opt_mutual_(false)
      , nb_total(0)
      , nb_before(0)
      , nb_after(0)
      , nleaves(0)
      , ncells(0)
      , time_tree_all(0)
      , time_tree_sample(0)
      , time_tree_exchange(0)
      , time_tree_growlocal(0)
      , time_tree_growglobal(0)
      , time_let_all      (0)
      , time_let_traverse (0)
      , time_let_req_all  (0)
      , time_let_req_comm (0)
      , time_let_res_all  (0)
      , time_let_res_attr_comm (0)
      , time_let_res_body_comm (0)
      , time_let_register (0)
      , time_map2_all(0)
      , time_map2_let(0)
      , time_map2_net(0)
#ifdef __CUDACC__
      , time_map2_dev(0)
#endif
      , let_func_count()
  {
    ReadEnv();
  }
  SharedData(const SharedData<TSP, SFC>& rhs) = delete; // no copy
  SharedData(SharedData<TSP, SFC>&& rhs) = delete; // no move

  bool SetOptMutual(bool b) {
    bool old = opt_mutual_;
    mapper_.opt_mutual_ = b;
    opt_mutual_ = b;
    return old;
  }

  /**
   * \brief Read some of the parameters from environmental variables
   */
  void ReadEnv() {
    const char *var = nullptr;
    if ((var = getenv("TAPAS_TASK_SPAWN_THRESHOLD"))) {
      opt_task_spawn_threshold_ = atoi(var);
      if (opt_task_spawn_threshold_ <= 0) {
        std::cout << "TAPAS_TASK_SPAWN_THRESHOLD must be > 0" << std::endl;
        exit(-1);
      }
    }
  }
};


} // namespace hot
} // namespace tapas

#endif // TAPAS_HOT_DATA_H_

