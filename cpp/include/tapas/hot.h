/**
 * @file hot.h
 * @brief Implements MPI-based, SFC (Space filling curves)-based HOT (Hashed Octree) implementation of Tapas
 */
#ifndef TAPAS_HOT_
#define TAPAS_HOT_

#include "tapas/stdcbug.h"

#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <memory>
#include <numeric>
#include <list>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <fstream>
#include <iomanip>
#include <iterator> // for std::back_inserter
#include <functional>
#include <limits>
#include <mutex>
#include <tuple>
#include <set>

#include <unistd.h>
#include <mpi.h>

#include "tapas/cell.h"
#include "tapas/bitarith.h"
#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/sfc_morton.h"
#include "tapas/threading/default.h"
#include "tapas/mpi_util.h"

#include "tapas/hot/let.h"

#define DEBUG_SENDRECV

using tapas::debug::BarrierExec;

namespace {
namespace iter = tapas::iterator;
}

namespace tapas {

/**
 * @brief Provides MPI-based distributed SFC-based octree partitioning
 */
namespace hot {

using tapas::mpi::MPI_DatatypeTraits;

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
  std::vector<index_t> leaf_nb_;   //!< Number of bodies in each cell
  std::vector<int>     leaf_owners_; //!< Owner process of leaf[i]
  std::vector<BodyType> local_bodies_; //!< Bodies that belong to the local process
  std::vector<KeyType>  local_body_keys_; //!< SFC keys of local bodies
  std::vector<BodyAttrType> local_body_attrs_; //!< Local body attributes

  std::vector<BodyType> let_bodies_;
  std::vector<BodyAttrType> let_body_attrs_;
  
  std::vector<KeyType> proc_first_keys_; //!< first SFC key of each process
  
  HotData() { }
  HotData(const HotData<TSP, SFC>& rhs) = delete; // no copy
  HotData(HotData<TSP, SFC>&& rhs) = delete; // no move
};

/**
 * @brief Remove redundunt elements in a std::vector. The vector must be sorted.
 * 
 * This way is much faster than using std::set.
 */
template<class T, class Iterator>
std::vector<T> uniq(Iterator beg, Iterator end) {
  std::vector<T> v(beg, end);
  v.erase(unique(std::begin(v), std::end(v)), std::end(v));
  return v;
}

/**
 * @brief Returns the range of bodies from an array of T (body type) that belong to the cell specified by the given key. 
 * @tparam BT Body type. (might be replaced by Iter::value_type)
 * @tparam Iter Iterator type of the body array.
 * @tparam Functor Functor type that retrieves morton key from a body type value.
 * @return returns std::pair of (pos, len)
 */
template <class SFC, class BT, class Iter, class Functor>
std::pair<typename SFC::KeyType, typename SFC::KeyType>
GetBodyRange(const typename SFC::KeyType k,
             Iter beg, Iter end,
             Functor get_key = __id<BT>) {
  using KeyType = typename SFC::KeyType;
  
  // When used in Refine(), a cells has sometimes no body.
  // In this special case, just returns (0, 0)
  if (beg == end) return std::make_pair(0, 0);
  
  auto less_than = [get_key] (const BT &hn, KeyType k) {
    return get_key(hn) < k;
  };
  
  auto fst = std::lower_bound(beg, end, k, less_than); // first node 
  auto lst = std::lower_bound(fst, end, SFC::GetNext(k), less_than); // last node
  
  assert(lst <= end);
  
  return std::make_pair(fst - beg, lst - fst); // returns (pos, nb)
}

/**
 * @brief std::vector version of GetBodyRange
 */
template<class SFC, class T, class Functor>
std::pair<typename SFC::KeyType, typename SFC::KeyType>
GetBodyRange(const typename SFC::KeyType k,
             const std::vector<T> &hn,
             Functor get_key = __id<T>) {
  using Iter = typename std::vector<T>::const_iterator;
  return GetBodyRange<SFC, T, Iter, Functor>(k, hn.begin(), hn.end(), get_key);
}
template <class TSP>
struct HelperNode {
  using KeyType = typename TSP::SFC::KeyType;
  KeyType key;          //!< SFC key (Default: Morton)
  Vec<TSP::Dim, int> anchor; //!< SFC key-like vector without depth information
  index_t p_index;      //!< Index of the corresponding body
  index_t np;           //!< Number of particles in a node
};

template<class T1, class T2>
static void Dump(const T1 &bodies, const T2 &keys, std::ostream & strm) {
#if 0
  // Tentatively disabled for BH. b.X[] is ExaFMM-specific member variables.
  for (size_t i = 0; i < bodies.size(); i++) {
    auto &b = bodies[i];
    strm << std::scientific << std::showpos << b.X[0] << " "
         << std::scientific << std::showpos << b.X[1] << " "
         << std::scientific << std::showpos << b.X[2] << " "
         << std::fixed << std::setw(10) << keys[i]
         << std::endl;
  }
#else
  // hack to avoid 'unused parameter' warnings
  (void) bodies;
  (void) keys;
  (void) strm;
#endif
}

template<class T1, class T2>
static void DumpToFile(const T1 &bodies,
                       const T2 &keys,
                       const std::string &fname,
                       bool append=false) {
  std::ios_base::openmode mode = std::ios_base::out;
  if (append) mode |= std::ios_base::app;
  std::ofstream ofs;
  ofs.open(fname.c_str(), mode);

  assert(ofs.good());
  Dump(bodies, keys, ofs);
  ofs.close();
}

template <class TSP>
std::vector<HelperNode<TSP>>
CreateInitialNodes(const typename TSP::BT::type *p, index_t np,
                   const Region<TSP> &r);

template <int DIM, class KeyType, class T>
void AppendChildren(KeyType k, T &s);

template <class TSP>
void SortBodies(const typename TSP::BT::type *b, typename TSP::BT::type *sorted,
                const HelperNode<TSP> *nodes,
                tapas::index_t nb);

template <class TSP>
void CompleteRegion(typename TSP::SFC x, typename TSP::SFC y, typename TSP::KeyVector &s);

template <class TSP>
index_t GetBodyNumber(const typename TSP::SFC k, const HelperNode<TSP> *hn,
                      index_t offset, index_t len);


template <class TSP> class Partitioner;

template<class TSP>
void FindLocalRoots(typename Cell<TSP>::KeyType,
                    const typename Cell<TSP>::CellHashTable&,
                    typename Cell<TSP>::KeySet&);

template <class TSP> // TapasStaticParams
class Cell: public tapas::BasicCell<TSP> {
  friend class Partitioner<TSP>;
  friend class iter::BodyIterator<Cell>;

  friend struct LET<TSP>;

  //========================================================
  // Typedefs 
  //========================================================
 public: // public type usings
  static const constexpr int Dim = TSP::Dim;
  typedef typename TSP::SFC SFC;
  typedef typename SFC::KeyType KeyType;
  
  typedef std::unordered_map<KeyType, Cell*> CellHashTable;
  using KeySet = std::unordered_set<KeyType>;
  
  typedef typename TSP::ATTR attr_type;
  typedef typename TSP::ATTR AttrType;
  typedef typename TSP::BT::type BodyType;
  typedef typename TSP::BT_ATTR BodyAttrType;
  typedef typename TSP::Threading Threading;

  using BodyIterator = iter::BodyIterator<Cell>;
  using SubCellIterator = iter::SubCellIterator<Cell>;
  
  using FP = typename TSP::FP;
  
  using Data = HotData<TSP, SFC>;

  template<class T>
  using VecPtr = std::shared_ptr<std::vector<T>>;

  friend void FindLocalRoots<TSP>(KeyType, const CellHashTable&, KeySet&);

  //========================================================
  // Constructors
  //========================================================
 public:

  static Cell *CreateLocalCell(KeyType k, std::shared_ptr<Data> data) {
    auto reg = CalcRegion(k, data->region_);

    // Check if I'm a leaf
    bool is_leaf = find(data->leaf_keys_.begin(), data->leaf_keys_.end(), k)
                   != data->leaf_keys_.end();

    int body_num, body_beg;

    if (is_leaf) {
      index_t beg, end;
      SFC::FindRangeByKey(data->local_body_keys_, k, beg, end);
      body_num = end - beg;
      body_beg = beg;
    } else {
      body_num = 0;
      body_beg = 0;
    }

    Cell *c = new Cell(reg, body_beg, body_num);
    c->key_ = k;
    c->is_leaf_ = is_leaf;
    c->is_local_ = true;
    c->is_local_subtree_ = false;
    c->data_ = data;
    c->bid_ = body_beg;
    c->nb_ = body_num;
    bzero(&c->attr_, sizeof(c->attr_));

    TAPAS_ASSERT(body_num >= 0);
    TAPAS_ASSERT(body_beg >= 0);
    
    return c;
  }

  static Cell *CreateRemoteCell(KeyType k, int nb, std::shared_ptr<Data> data) {
    auto reg = CalcRegion(k, data->region_);

    Cell *c = new Cell(reg, 0, 0);
    c->key_ = k;
    c->is_leaf_ = false;
    c->is_local_ = false;
    c->is_local_subtree_ = false;
    c->nb_ = nb;
    c->remote_bodies_.clear();
    c->remote_body_attrs_.resize(nb);
    c->data_ = data;
    bzero(&c->attr_, sizeof(c->attr_));
    
    return c;
  }

  Cell(const Region<TSP> &region, index_t bid, index_t nb)
      : tapas::BasicCell<TSP>(region, bid, nb) {} 

  //========================================================
  // Member functions
  //========================================================

 public:
  KeyType key() const { return key_; }
  
  bool operator==(const Cell &c) const;
  template <class T>
  bool operator==(const T &) const { return false; }
  bool IsRoot() const;
  bool IsLocalSubtree() const;

  // 1-parameter Map function
  template <class Funct>
  static void Map(Funct f, Cell<TSP> &c);

  // 2-argument Map() function
  template<class Funct>
  static void Map(Funct f, Cell<TSP> &c1, Cell<TSP> &c2);

  static void PostOrderMap(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f);
  static void UpwardMap(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f);
  
  static void Map(BodyIterator &b1, BodyIterator &b2,
                  std::function<void(BodyIterator&, BodyIterator&)> f) {
    f(b1, b2);
  }

  static void Map(BodyIterator &b1,
                  std::function<void(BodyIterator)> f) {
    f(b1);
  }

  /**
   * @brief Returns if the cell is a leaf cell
   */
  bool IsLeaf() const;
  void SetLeaf(bool);

  /**
   * @brief Returns if the cell is local.
   */
  bool IsLocal() const;

  /**
   * @brief Returns the number of subcells. This is 0 or 2^DIM in HOT algorithm.
   */
  int nsubcells() const;
  
  /**
   * @brief Returns idx-th subcell.
   */
  Cell &subcell(int idx);

  /**
   * @brief Returns the parent cell if it's local.
   *
   * Returns a reference to the parent cell object of this cell.
   * In this HOT implementation, parent cell of a local cell is
   * always a local cell.
   */
  Cell &parent() const;

  int depth() const {
    return SFC::GetDepth(key_);
  }

  Data &data() { return *data_; }
  std::shared_ptr<Data> data_ptr() { return data_; }
  
#ifdef DEPRECATED
  typename TSP::BT::type &particle(index_t idx) const {
    return body(idx);
  }
#endif

  // Accessor functions to bodies & body attributes
  BodyType &body(index_t idx);
  const BodyType &body(index_t idx) const;

  BodyType &local_body(index_t idx);
  const BodyType &local_body(index_t idx) const;
  
  BodyIterator bodies() {
    return BodyIterator(*this);
  }

  BodyAttrType &body_attr(index_t idx);
  const BodyAttrType &body_attr(index_t idx) const;
  
  BodyAttrType *body_attrs();
  const BodyAttrType *body_attrs() const;
  
  BodyAttrType &local_body_attr(index_t idx);
  const BodyAttrType &local_body_attr(index_t idx) const;
  
  BodyAttrType *local_body_attrs();
  const BodyAttrType *local_body_attrs() const;
  
  /**
   * \brief Get number of local particles.
   * This function is mainly for debugging or checking result.
   * It is not recommended to use local_nb() for your main computation.
   * because it exposes the underlying implementation details of Tapas runtime.
   */ 
  size_t local_nb() const {
    return (size_t) data_->local_bodies_.size();
  }

  size_t nb() const {
    if (!this->IsLeaf()) {
      TAPAS_ASSERT(!"Cell::nb() is not allowed for non-leaf cells.");
    }

    return nb_;
  }

  static Region<TSP>  CalcRegion(KeyType, const Region<TSP>& r);
  static tapas::Vec<Dim, FP> CalcCenter(KeyType, const Region<TSP>& r);
  
#ifdef DEPRECATED
  typename TSP::BT_ATTR *particle_attrs() const {
    return body_attrs();
  }
#endif
  SubCellIterator subcells() {
    return SubCellIterator(*this);
  }

  const Region<TSP> &region() const { return data_->region_; }

 protected:
  // utility/accessor functions
  Cell *Lookup(KeyType k) const;
  CellHashTable *ht() { return ht_; }
  virtual void make_pure_virtual() const {}
  void RegisterCell(Cell<TSP> *c);

  //========================================================
  // Member variables
  //========================================================
 protected:
  KeyType key_; //!< Key of the cell
  bool is_leaf_;
  std::shared_ptr<Data> data_;
  
  int nb_; //!< number of bodies in the local process (not bodies under this cell).
  
  std::shared_ptr<CellHashTable> ht_; //!< Hash table of KeyType -> Cell*
  std::shared_ptr<std::mutex>    ht_mtx_; //!< mutex to manipulate ht_
  
  bool is_local_; //!< if it's a local cell or LET cell.
  bool is_local_subtree_; //!< If all of its descendants are local.
  std::vector<BodyType> remote_bodies_; //!< LET bodies (if !is_local_)
  std::vector<BodyAttrType> remote_body_attrs_; //!< LET body attrs (If !is_local_)
}; // class Cell

template<class T>
using uset = std::unordered_set<T>;

#ifdef TAPAS_BH

template<class TSP>
std::string k2s(typename Cell<TSP>::KeyType k) {
  std::stringstream ss;
  using SFC = typename Cell<TSP>::SFC;
#if 0
  ss << SFC::Simplify(k) << " " << SFC::Decode(k);
#else
  ss << SFC::Simplify(k);
#endif
  return ss.str();
}

// new Traverse
template<class TSP>
void ReportSplitType(typename Cell<TSP>::KeyType trg_key,
                     typename Cell<TSP>::KeyType src_key,
                     SplitType by_pred, SplitType orig) {
  using SFC = typename Cell<TSP>::SFC;
  
  Stderr e("check");
  e.out() << SFC::Simplify(trg_key) << " - " << SFC::Simplify(src_key) << "  ";

  e.out() << "Pred:";
  switch(by_pred) {
    case SplitType::Approx:     e.out() << "Approx";     break;
    case SplitType::Body:       e.out() << "Body";       break;
    case SplitType::SplitLeft:  e.out() << "SplitLeft";  break;
    case SplitType::SplitRight: e.out() << "SplitRight"; break;
    case SplitType::None:       e.out() << "None:";      break;
    default: assert(0);
  }

  e.out() << " Orig:";
  switch(orig) {
    case SplitType::Approx:     e.out() << "Approx";     break;
    case SplitType::Body:       e.out() << "Body";       break;
    case SplitType::SplitLeft:  e.out() << "SplitLeft";  break;
    case SplitType::SplitRight: e.out() << "SplitRight"; break;
    case SplitType::None:       e.out() << "None:";      break;
    default: assert(0);
  }
  
  e.out() << " " << (by_pred == orig ? "OK" : "NG") << std::endl;
}


// new Traverse

#endif // TAPAS_BH


template<class T>
int MPI_Allreduce(const std::vector<T> &send, std::vector<T> &recv, MPI_Op op, MPI_Comm comm) {
  recv.resize(send.size());
  return ::MPI_Allreduce(static_cast<const T*>(send.data()),
                         static_cast<T*>(recv.data()),
                         (int)send.size(),
                         MPI_DatatypeTraits<T>::type(),
                         op, comm);
}

// Utility functions on set-related operations on std::vector

/**
 * @brief Returns difference of two sets (a simple wrapper of std::set_difference)
 *
 * Returns an instance of the container type C which contains elements that are
 * in c1 but not c2. c1 and c2 must be sorted.
 */
template<class C>
C SetDiff(const C& c1, const C& c2) {
  C res;
  std::set_difference(std::begin(c1), std::end(c1),
                      std::begin(c2), std::end(c2),
                      std::back_inserter(res));
  return res;
}

#if 0
/**
 * @brief Returns union of two sets (a simple wrapper of std::set_union)
 *
 * c1 and c2 must be sorted.
 */
template<class C>
C SetUnion(const C& c1, const C& c2) {
  C res;
  std::set_union(std::begin(c1), std::end(c1),
                 std::begin(c2), std::end(c2),
                 std::back_inserter(res));
  return res;
}
#endif


/**
 * @brief Return a new Region object that covers all Regions across multiple MPI processes
 */
template<class TSP>
Region<TSP> ExchangeRegion(const Region<TSP> &r) {
  const int Dim = TSP::Dim;
  typedef typename TSP::FP FP;

  Vec<Dim, FP> new_max, new_min;

  // Exchange max
  ::MPI_Allreduce(&r.max()[0], &new_max[0], Dim, MPI_DatatypeTraits<FP>::type(), MPI_MAX, MPI_COMM_WORLD);
  
  // Exchange min
  ::MPI_Allreduce(&r.min()[0], &new_min[0], Dim, MPI_DatatypeTraits<FP>::type(), MPI_MIN, MPI_COMM_WORLD);

  return Region<TSP>(new_min, new_max);
}

/**
 * @brief Create an array of HelperNode from bodies
 * In the first stage of tree construction, one HelperNode is create for each body.
 * @return Array of HelperNode
 * @param bodies Pointer to an array of bodies
 * @param nb Number of bodies (length of bodies)
 * @param r Region object
 */
template <class TSP>
std::vector<HelperNode<TSP>> CreateInitialNodes(const typename TSP::BT::type *bodies,
                                                     index_t nb,
                                                     const Region<TSP> &r) {
    const int Dim = TSP::Dim;
    typedef typename TSP::FP FP;
    typedef typename TSP::BT BT;
    typedef typename TSP::SFC SFC;
    typedef HelperNode<TSP> HN;

    std::vector<HN> nodes(nb);
    FP num_cell = 1 << SFC::MAX_DEPTH; // maximum number of cells in one dimension
    Vec<Dim, FP> pitch;           // possible minimum cell width
    for (int d = 0; d < Dim; ++d) {
      pitch[d] = (r.max()[d] - r.min()[d]) / num_cell;
    }

    for (index_t i = 0; i < nb; ++i) {
        // First, create 1 helper cell per particle
        HN &node = nodes[i];
        node.p_index = i;
        node.np = 1;

        // Particle pos offset is the offset of each coordinate value (x,y,z) in body structure
        Vec<Dim, FP> off = ParticlePosOffset<Dim, FP, BT::pos_offset>::vec((const void*)&(bodies[i]));
        off -= r.min(); // set the base 0
        off /= pitch;   // quantitize offsets

        // Now 'off' is a Dim-dimensional index of the finest-level cell to which the particle belong.
        for (int d = 0; d < Dim; ++d) {
            node.anchor[d] = (int)(off[d]);
            // assume maximum boundary is inclusive, i.e., a particle can be
            // right at the maximum boundary.
            if (node.anchor[d] == (1 << SFC::MAX_DEPTH)) {
                TAPAS_LOG_DEBUG() << "Particle located at max boundary." << std::endl;
                node.anchor[d]--;
            }
        }
#ifdef TAPAS_DEBUG
        TAPAS_ASSERT(node.anchor >= 0);
# if 1
        if (!(node.anchor < (1 << SFC::MAX_DEPTH))) {
            TAPAS_LOG_ERROR() << "Anchor, " << node.anchor
                              << ", exceeds the maximum depth." << std::endl
                              << "Particle at "
                              << ParticlePosOffset<Dim, FP, BT::pos_offset>::vec((const void*)&(bodies[i]))
                              << std::endl;
            TAPAS_DIE();
        }
# else
        assert(node.anchor < (1 << SFC::MAX_DEPTH));
# endif
#endif // TAPAS_DEBUG

        node.key = SFC::CalcFinestKey(node.anchor);
    }

    return nodes;
}

template <class TSP>
void SortBodies(const typename TSP::BT::type *b, typename TSP::BT::type *sorted,
                const HelperNode<TSP> *sorted_nodes,
                tapas::index_t nb) {
    for (index_t i = 0; i < nb; ++i) {
        sorted[i] = b[sorted_nodes[i].p_index];
    }
}

template <int DIM, class SFC, class T>
void AppendChildren(typename SFC::KeyType x, T &s) {
  using KeyType = typename SFC::KeyType;
  
  int x_depth = SFC::GetDepth(x);
  int c_depth = x_depth + 1;
  if (c_depth > SFC::MAX_DEPTH) return;
  x = SFC::IncrementDepth(x, 1);
  for (int i = 0; i < (1 << DIM); ++i) {
    KeyType child_key = ((KeyType)i << ((KeyType::MAX_DEPTH - c_depth) * DIM +
                                        SFC::DEPTH_BIT_WIDTH));
    s.push_back(x | child_key);
    TAPAS_LOG_DEBUG() << "Adding child " << (x | child_key) << std::endl;
  }
}

template <class TSP>
void CompleteRegion(typename TSP::SFC::KeyType x,
                    typename TSP::SFC::KeyType y,
                    typename TSP::SFC::KeyVector &s) {
  typedef typename TSP::SFC SFC;
  typedef typename SFC::KeyType KeyType;
  
  KeyType fa = SFC::FinestAncestor(x, y);
  typename SFC::KeyList w;
  
  AppendChildren<TSP::Dim, KeyType>(fa, w);
  tapas::PrintKeys(w, std::cout);
  
  while (w.size() > 0) {
    KeyType k = w.front();
    w.pop_front();
    TAPAS_LOG_DEBUG() << "visiting " << k << std::endl;
    if ((k > x && k < y) && !SFC::IsDescendant(k, y)) {
      s.push_back(k);
      TAPAS_LOG_DEBUG() << "Adding " << k << " to output set" << std::endl;
    } else if (SFC::IsDescendant(k, x) || SFC::IsDescendant(k, y)) {
      TAPAS_LOG_DEBUG() << "Adding children of " << k << " to work set" << std::endl;
      AppendChildren<TSP>(k, w);
    }
  }
  std::sort(std::begin(s), std::end(s));
}

/**
 * @brief 1-parameter Map function for (deprecated)
 */
template <class TSP>
template <class Funct>
void Cell<TSP>::Map(Funct f, Cell<TSP> &cell) {
  f(cell);
}

/**
 * 2-parameter Map function over cells (apply user function to products of cells)
 */
template<class TSP>
template<class Funct>
void Cell<TSP>::Map(Funct f, Cell<TSP> &c1, Cell<TSP> &c2) {
#ifdef TAPAS_BH
  if (c1.key() == 0 && c2.key() == 0) {
    LET<TSP>::Exchange(f, c1);
  }
#endif
  
  f(c1, c2);
}

/**
 * \brief PostOrderMap starting from a local cell. The subtree under c must be completely local.
 */
template<class TSP>
void LocalUpwardTraversal(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f) {
  if (!c.IsLocal()) {
    using SFC = typename Cell<TSP>::SFC;
    auto k = c.key();
    std::cerr << SFC::Simplify(k) << " "
              << SFC::Decode(k) << " "
              << k << std::endl;
  }
  TAPAS_ASSERT(c.IsLocal());
      
  if (c.IsLeaf()) {
    f(c);
  } else {
    int nc = c.nsubcells();
    for (int ci = 0; ci < nc; ci++) {
      Cell<TSP> &child = c.subcell(ci);
      LocalUpwardTraversal(child, f);
    }
    f(c);
  }
}

/**
 * \brief Exchange cell attrs of global leaves
 */
template<class TSP>
void ExchangeGlobalLeafAttrs(typename Cell<TSP>::CellHashTable &gtree,
                             const typename Cell<TSP>::KeySet &lroots) {
  // data.gleaves_ is unnecessary?
  using KeyType = typename Cell<TSP>::KeyType;
  using AttrType = typename Cell<TSP>::AttrType;
  
  std::vector<KeyType> keys_send(lroots.begin(), lroots.end());
  std::vector<KeyType> keys_recv;
  std::vector<AttrType> attr_send;
  std::vector<AttrType> attr_recv;
  
  auto &data = gtree[0]->data();

  for(size_t i = 0; i < keys_send.size(); i++) {
    KeyType k = keys_send[i];
    TAPAS_ASSERT(data.ht_.count(k) == 1);
    attr_send.push_back(data.ht_[k]->attr());
  }

  tapas::mpi::Allgatherv(keys_send, keys_recv, MPI_COMM_WORLD);
  tapas::mpi::Allgatherv(attr_send, attr_recv, MPI_COMM_WORLD);

  for (size_t i = 0; i < keys_recv.size(); i++) {
    KeyType key = keys_recv[i];
    TAPAS_ASSERT(gtree.count(key) == 1);
    gtree[key]->attr() = attr_recv[i];
  }

  TAPAS_ASSERT(keys_recv.size() == attr_recv.size());
}

template<class TSP>
void GlobalUpwardTraversal(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f) {
  using KeyType = typename Cell<TSP>::KeyType;
  using SFC = typename Cell<TSP>::SFC;
  auto &data = c.data();
  KeyType k = c.key();

  // c must be in the global tree hash table.
  TAPAS_ASSERT(data.ht_gtree_.count(k) == 1);

  // There are only two cases:
  // 1.  All the children of the cell c are in the global tree.
  //     This means c is not global-leaf cell.
  // 2.  None of the children of the cell c is in the global tree.
  //     This means c is a global-leaf.

  if (data.gleaves_.count(k) > 0) {
    // the cell c is a global leaf. The attr value is already calculated
    // as a local root in its owner process.
    return;
  }

  TAPAS_ASSERT(data.ht_gtree_.count(k) > 0);
  
  // c is not a global leaf.
  int nc = c.nsubcells();
  for (int ci = 0; ci < nc; ci++) {
    KeyType chk = SFC::Child(c.key(), ci);
    Cell<TSP> *child = data.ht_gtree_.at(chk);
    GlobalUpwardTraversal(*child, f);
  }
  
  f(c);
}

template<class TSP>
void Cell<TSP>::PostOrderMap(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f) {
  auto &data = c.data();
  
  // perform post-order (upward) traverse
  // algorithm:
  //   if c is in the global tree:
  //     for lr in local roots:
  //       perform local upward up to lr
  //     end for
  //     exchange global leaves
  //     perform global upward up to c
  //   else
  //     perform local upward up to c
  //   end if

  
  if (data.ht_gtree_.count(c.key()) > 0) {
    // c is in the global tree. We need to
    // 1. calculate local roots in each process,
    // 2. allgatherv the results
    // 3. culculate the global tree in each process

    for (auto && key_lr : data.lroots_) {
      TAPAS_ASSERT(data.ht_.count(key_lr) == 1);
      auto *cell = data.ht_[key_lr];
      LocalUpwardTraversal(*cell, f);
    }

    ExchangeGlobalLeafAttrs<TSP>(data.ht_gtree_, data.lroots_);

    GlobalUpwardTraversal(c, f);
  } else {
    
    // c is not in the global tree, which means c's subtree is perfectly local.
    // This means that the user called PostOrderMap not from the root cell.
    // We're not sure yet if such invocation is allowed in Tapas programming model.
    
    assert(false); // for debug
    LocalUpwardTraversal(c, f);
  }
}

template<class TSP>
void Cell<TSP>::UpwardMap(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f) {
  Cell<TSP>::PostOrderMap(c, f);
}

template <class TSP>
bool Cell<TSP>::operator==(const Cell &c) const {
  return key_ == c.key_;
}

template <class TSP>
bool Cell<TSP>::IsRoot() const {
  return SFC::GetDepth(key_) == 0;
}

template <class TSP>
bool Cell<TSP>::IsLocalSubtree() const {
  return is_local_subtree_;
}


template <class TSP>
bool Cell<TSP>::IsLeaf() const {
  return is_leaf_;
}

template <class TSP>
void Cell<TSP>::SetLeaf(bool b) {
  is_leaf_ = b;
}

template <class TSP>
bool Cell<TSP>::IsLocal() const {
  return is_local_;
}

template <class TSP>
int Cell<TSP>::nsubcells() const {
  if (IsLeaf()) return 0;
  else return (1 << TSP::Dim);
}

template <class TSP>
Cell<TSP> &Cell<TSP>::subcell(int idx) {
  if (IsLeaf()) {
    TAPAS_LOG_ERROR() << "Trying to access children of a leaf cell." << std::endl;
    TAPAS_DIE();
  }

  KeyType child_key = SFC::Child(key_, idx);
  Cell *c = Lookup(child_key);

  if (c == nullptr) {
#if 0
    //\todo: fix memory leak
    // (use pseudo leaf-only hash table)
    //c = new Cell<TSP>(child_key, false, 0, 0, data_);
    c = new Cell<TSP>(child_key, data_); // create a dummy cell
#else
    std::stringstream ss;
    ss << "In MPI rank " << data_->mpi_rank_ << ": " 
       << "Cell not found for key "
       << child_key << " "
       << SFC::Simplify(child_key) << " "
       << SFC::Decode(child_key) << std::endl;
    ss << "In MPI rank " << data_->mpi_rank_ << ": Anscestors are:" << std::endl;

    for (KeyType k = key_; k != 0; k = SFC::Parent(k)) {
      ss << "      "
         << SFC::Simplify(k) << " "
         << SFC::Decode(k) << " "
         << k << std::endl;
    }
    
    TAPAS_LOG_ERROR() << ss.str(); abort();
    TAPAS_ASSERT(c != nullptr);
#endif
  }
  
  return *c;
}

template <class TSP>
Cell<TSP> *Cell<TSP>::Lookup(KeyType k) const {
  // Try the local hash.
  auto &ht = data_->ht_;
  auto &ht_let = data_->ht_let_;
  auto &ht_gtree = data_->ht_gtree_;
  
  auto i = ht.find(k);
  if (i != ht.end()) {
    assert(i->second != nullptr);
    return i->second;
  }
  
  i = ht_let.find(k);
  // If the key is not in local hash, next try LET hash.
  if (i != ht_let.end()) {
    assert(i->second != nullptr);
    return i->second;
  }

  // NOTE: we do not search in ht_gtree.
  //       ht_gtree is for global tree, and used only in upward phase.
  //       If a cell exists only in ht_gtree, only the value of attr of the cell matters
  //       and whether the cell is leaf or not.
  //       Thus, cells in ht_gtree are not used in dual tree traversal.
  i = ht_gtree.find(k);
  // If the key is not in local hash, next try LET hash.
  if (i != ht_gtree.end()) {
    assert(i->second != nullptr);
    return i->second;
  }

  
  return nullptr;
}

template <class TSP>
Cell<TSP> &Cell<TSP>::parent() const {
    if (IsRoot()) {
        TAPAS_LOG_ERROR() << "Trying to access parent of the root cell." << std::endl;
        TAPAS_DIE();
    }
    KeyType parent_key = SFC::Parent(key_);
    auto *c = Lookup(parent_key);
    if (c == nullptr) {
      TAPAS_LOG_ERROR() << "Parent (" << parent_key << ") of "
                        << "cell (" << key_ << ") not found.\n"
                        << "Parent key = " << SFC::Decode(parent_key) << "\n"
                        << "Child key =  " << SFC::Decode(key_)
                        << std::endl;
      TAPAS_DIE();
    }
    return *c;
}

template <class TSP>
const typename TSP::BT::type &Cell<TSP>::body(index_t idx) const {
  TAPAS_ASSERT(this->IsLeaf() && "Cell::body(...) is not allowed for non-leaf cells.");
  TAPAS_ASSERT(this->nb() >= 0);
  TAPAS_ASSERT((size_t)idx < this->nb());
  
  if (is_local_) {
    return data_->local_bodies_[this->bid() + idx];
  } else {
    return remote_bodies_[idx];
  }
}

template <class TSP>
typename TSP::BT::type &Cell<TSP>::body(index_t idx) {
  return const_cast<typename TSP::BT::type &>(const_cast<const Cell<TSP>*>(this)->body(idx));
}

template <class TSP>
const typename TSP::BT::type &Cell<TSP>::local_body(index_t idx) const {
  TAPAS_ASSERT(idx < (index_t)data_->local_bodies_.size());
  TAPAS_ASSERT(this->IsLocal() && "Cell::local_body() can be called only for local cells.");
  
  return data_->local_bodies_[this->bid() + idx];
}

template <class TSP>
typename TSP::BT::type &Cell<TSP>::local_body(index_t idx) {
  return const_cast<typename TSP::BT::type &>(const_cast<const Cell<TSP>*>(this)->local_body(idx));
}

template <class TSP>
const typename TSP::BT_ATTR *Cell<TSP>::body_attrs() const {
  TAPAS_ASSERT(this->IsLeaf() && "Cell::body_attrs(...) is not allowed for non-leaf cells.");

  if (is_local_) {
    return data_->local_body_attrs_.data() + this->bid();
  } else {
    return remote_body_attrs_.data();
  }
}

template <class TSP>
typename TSP::BT_ATTR *Cell<TSP>::body_attrs() {
  return const_cast<typename TSP::BT_ATTR &>(const_cast<const Cell<TSP>*>(this)->local_attrs());
}

template <class TSP>
const typename TSP::BT_ATTR &Cell<TSP>::body_attr(index_t idx) const {
  TAPAS_ASSERT(this->IsLeaf() && "Cell::body_attr(...) is not allowed for non-leaf cells.");
  TAPAS_ASSERT(idx < (index_t)this->nb());
  
  if (is_local_) {
    return this->data_->local_body_attrs_[this->bid() + idx];
  } else {
    TAPAS_ASSERT((size_t)this->nb() == remote_body_attrs_.size());
    return remote_body_attrs_[idx];
  }
}

template <class TSP>
typename TSP::BT_ATTR &Cell<TSP>::body_attr(index_t idx) {
  return const_cast<typename TSP::BT_ATTR &>(const_cast<const Cell<TSP>*>(this)->body_attr(idx));
}

/**
 * \brief Returns a pointer to the first element of local bodies.
 * This function breaks the abstraction of Tapas and should be used only for 
 * debugging / result checking purpose.
 */
template <class TSP>
const typename TSP::BT_ATTR *Cell<TSP>::local_body_attrs() const {
  TAPAS_ASSERT(this->IsLocal() && "Cell::local_body_attrs() is only allowed for local cells");
  
  return data_->local_body_attrs_.data();
}

/**
 * \brief Non-const version of local_body_attrs()
 */
template <class TSP>
typename TSP::BT_ATTR *Cell<TSP>::local_body_attrs() {
  return const_cast<typename TSP::BT_ATTR *>(const_cast<const Cell<TSP>*>(this)->local_body_attrs());
}


/**
 * \brief Returns an attr of a body specified by idx.
 * This function breaks the abstraction of Tapas, thus should be used only for debugging purpose.
 */
template <class TSP>
const typename TSP::BT_ATTR &Cell<TSP>::local_body_attr(index_t idx) const {
  TAPAS_ASSERT(this->IsLocal() && "Cell::local_body_attr(...) is allowed only for local cells.");
  TAPAS_ASSERT(idx < (index_t)data_->local_body_attrs_.size());
  
  return data_->local_body_attrs_[this->bid() + idx];
}

/**
 * \brief Non-const version of Cell::local_body_attr()
 */
template <class TSP>
typename TSP::BT_ATTR &Cell<TSP>::local_body_attr(index_t idx) {
  return const_cast<typename TSP::BT_ATTR &>(const_cast<const Cell<TSP>*>(this)->local_body_attr(idx));
}

template <class TSP> // Tapas static params
class Partitioner {
 private:
  const int max_nb_;
  
  using BodyType = typename TSP::BT::type;
  using KeyType = typename Cell<TSP>::KeyType;
  using CellAttrType = typename Cell<TSP>::AttrType;
  using CellHashTable = typename Cell<TSP>::CellHashTable;
  
  using KeySet = typename Cell<TSP>::SFC::KeySet;
  
  using SFC = typename TSP::SFC;
  using HT = typename Cell<TSP>::CellHashTable;

  public:
    Partitioner(unsigned max_nb): max_nb_(max_nb) {}

    Cell<TSP> *Partition(typename TSP::BT::type *b, index_t nb,
                         const Region<TSP> &r);
    Cell<TSP> *Partition(std::vector<typename TSP::BT::type> &b,
                         const Region<TSP> &r);
 private:
#if 0
  void Refine(Cell<TSP> *c,
              const std::vector<HelperNode<TSP>> &hn,
              const BodyType *b,
              int cur_depth,
              KeyType cur_key) const;
#endif

 public:
  //---------------------
  // Supporting functions
  //---------------------
  
  /**
   * @brief Find owner process from a head-key list.
   * The argument head_list contains SFC keys that are the first keys of processes.
   * head_list[P] is the first SFC key belonging to process P.
   * Because the first element is always 0 (by definition of space filling curve),
   * the result must be always >= 0.
   * 
   */
  static int
  FindOwnerProcess(const std::vector<KeyType> &head_list, KeyType key) {
    TAPAS_ASSERT(Cell<TSP>::SFC::RemoveDepth(head_list[0]) == 0);
    auto comp = [](KeyType a, KeyType b) {
      return Cell<TSP>::SFC::RemoveDepth(a) < Cell<TSP>::SFC::RemoveDepth(b);
    };
    return std::upper_bound(head_list.begin(), head_list.end(), key, comp) - head_list.begin() - 1;
  }

  static std::vector<int>
  FindOwnerProcess(const std::vector<KeyType> &head_key_list,
                   const std::vector<KeyType> &keys) {
    std::vector<int> owners(keys.size());
    for (size_t i = 0; i < keys.size(); i++) {
      owners[i] = FindOwnerProcess(head_key_list, keys[i]);
    }

    return owners;
  }

  /**
   * \brief Select cells to be sent as a response in LET::Exchange
   * The request lists are made conservatively, thus not all the requested cells exist in the sender process.
   * Check the requested list and replace non-existing cells with existing cells by the their finest anscestors.
   * If attribute of a cell is requested but the cell is actually a leaf, 
   * both of the attribut and body must be sent.
   */
  static void SelectResponseCells(std::vector<KeyType> &attr_keys, std::vector<int> &attr_src_pids,
                                  std::vector<KeyType> &body_keys, std::vector<int> &body_src_pids,
                                  const HT& hash) {
    std::set<std::pair<int, KeyType>> res_attr; // keys (and their destinations) of which attributes will be sent as response.
    std::set<std::pair<int, KeyType>> res_body; // keys (and their destinations) of which bodies will be sent as response.

    for (size_t i = 0; i < attr_keys.size(); i++) {
      KeyType k = attr_keys[i];
      int src_pid = attr_src_pids[i]; // PID of the process that requested k.

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      if (k == 0) {
        // This should not happend because if k is root node,
        // that means the process does not have any anscestor of the original k (except the root).
        // The requester sent the request to a wrong process
        TAPAS_ASSERT(false);
      }

      res_attr.insert(std::make_pair(src_pid, k));

      if (hash.at(k)->IsLeaf()) {
        res_body.insert(std::make_pair(src_pid, k));
      }
    }
        
    for (size_t i = 0; i < body_keys.size(); i++) {
      KeyType k = body_keys[i];
      int src_pid = body_src_pids[i];

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      TAPAS_ASSERT(k != 0); // the same reason above
      TAPAS_ASSERT(hash.count(k) > 0);
      TAPAS_ASSERT(hash.at(k)->IsLeaf());

      res_body.insert(std::make_pair(src_pid, k));
    }

#if TAPAS_DEBUG
    BarrierExec([&res_attr, &res_body](int rank, int size) {
        std::cerr << "Rank " << rank << " SelectResponseCells: keys_attr.size() = " << res_attr.size() << std::endl;
        std::cerr << "Rank " << rank << " SelectResponseCells: keys.body.size() = " << res_body.size() << std::endl;
      });
#endif

    // Set values to the vectors
    attr_keys.resize(res_attr.size());
    attr_src_pids.resize(res_attr.size());

    int idx = 0;
    for (auto & iter : res_attr) {
      attr_src_pids[idx] = iter.first;
      attr_keys[idx] = iter.second;
      idx++;
    }

    body_keys.resize(res_body.size());
    body_src_pids.resize(res_body.size());

    idx = 0;
    for (auto & iter : res_body) {
      body_src_pids[idx] = iter.first;
      body_keys[idx] = iter.second;

      idx++;
    }

    return;
  }
  
  static void KeysToAttrs(const std::vector<KeyType> &keys,
                          std::vector<CellAttrType> &attrs,
                          const HT& hash) {
    auto key_to_attr = [&hash](KeyType k) -> CellAttrType {
      return hash.at(k)->attr();
    };
    attrs.resize(keys.size());
    std::transform(keys.begin(), keys.end(), attrs.begin(), key_to_attr);
  }

  static int FindOwnerByKey(const std::vector<KeyType> &leaf_keys,
                            const std::vector<int> &leaf_owners, KeyType key) {
    size_t at = std::lower_bound(leaf_keys.begin(), leaf_keys.end(), key) - leaf_keys.begin();
    assert(at < leaf_keys.size());

    return (int)leaf_owners[at];
  }
    
  static void KeysToBodies(const std::vector<KeyType> &keys,
                           std::vector<index_t> &nb,
                           std::vector<BodyType> &bodies,
                           const HT& hash) {
    bodies.resize(keys.size());
    nb.resize(keys.size());

    // In BH, each leaf has 0 or 1 body (while every cell has attribute)
    for (size_t i = 0; i < keys.size(); i++) {
      KeyType k = keys[i];
      auto *c = hash.at(k);
      nb[i] = c->IsLeaf() ? c->nb() : 0;
      if (nb[i] > 0) {
        bodies[i] = hash.at(k)->body(0);
      }
    }
  }

  /**
   * \brief Remove local cell keys from a KeySet
   * Used in ExchangeELT
   */
  void RemoveLocalKeys(KeySet keys, const CellHashTable &hash) {
    for (auto k : hash.keys()) {
      if (keys.count(k) > 0) {
        keys.remove(k);
      }
    }
  }
}; // class Partitioner

/**
 * \brief Create a list of keys where i-th element is the first key of process i.
 */
template<class KeyType>
std::vector<KeyType> ProcHeadKeys(std::vector<KeyType> leaf_keys,
                                  std::vector<int> &leaf_owners,
                                  int num_proc) {
  std::vector<KeyType> head_list(num_proc);
  
  // NOTE: Process 0's first key is always 0
  //TAPAS_ASSERT(Key::RemoveDepth(leaf_keys[0]) == (KeyType)0);
  
  int pos = 0;
  for (int p = 0; p < num_proc; p++) {
    head_list[p] = leaf_keys[pos];
    while(leaf_owners[pos] <= p) pos++;
  }
  return head_list;
}

/**
 * @brief Overloaded version of Partitioner::Partition
 */
template <class TSP>
Cell<TSP>*
Partitioner<TSP>::Partition(std::vector<typename TSP::BT::type> &b, const Region<TSP> &r) {
    return Partitioner<TSP>::Partition(b.data(), b.size(), r);
}

/**
 * @brief Split cells that have more than nb_max bodies (not recursive)
 *
 * @param cell_keys Array of morton keys of cells
 * @param nb Array of current numbers of bodies of the cells
 * @param max_nb Criteria to split a cell
 */
template<class TSP>
std::vector<typename TSP::SFC::KeyType>
SplitLargeCellsOnce(const std::vector<typename TSP::SFC::KeyType> &cell_keys,
                    const std::vector<index_t> &nb,
                    int max_nb, int *max_depth) {
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;
  
  std::vector<KeyType> ret; // new

  for (size_t i = 0; i < cell_keys.size(); i++) {
    if (nb[i] <= max_nb) {
      // This cell does not need to be split.
      ret.push_back(cell_keys[i]);
    } else {
      // Create 2^DIM children (8 in 3-dim)
      if (SFC::GetDepth(cell_keys[i]) >= SFC::MAX_DEPTH) {
        TAPAS_LOG_ERROR()
            << "Error: Reached maximum depth of octree. "
            << "Maybe some particles have the exact same coordinates?" << std::endl;
        MPI_Finalize();
        exit(-1);
      }
      
      auto children = SFC::GetChildren(cell_keys[i]);
      *max_depth = std::max(*max_depth, SFC::GetDepth(children[0]));
      for (auto ch : children) {
        ret.push_back(ch);
      }
    }
  }

  return ret;
}


/**
 * @brief Distribute cells over processes in a simple algorithm so that each process has
 *        roughly equal number of bodies
 *
 * @param nb Array of numbers of bodies in cells
 * @param proc_size Number of processes (assumes process numbers are integer starting from 0)
 * @return Array of process numbers
 * @todo inline is required here to avoid multiple definitions
 */
inline std::vector<int> SplitKeysSimple(const std::vector<index_t> &nb, int proc_size) {
  index_t total_nb = std::accumulate(nb.begin(), nb.end(), 0); // total number of bodies
  index_t guide = total_nb / proc_size;

  std::vector<int> ret(nb.size()); // return value
  int psum = 0; // partial sum
  int cur_proc = 0;

  for (size_t i = 0; i < nb.size(); i++) {
    psum += nb[i];
    ret[i] = cur_proc;
    if (psum > guide * (cur_proc+1)) {
      // remaining part belongs to last proc.
      cur_proc = std::min(cur_proc + 1, proc_size - 1);
    }
  }

  return ret;
}

template<class T>
size_t sum(const T& c) {
  return std::accumulate(c.begin(), c.end(), 0);
}

template<class TSP>
void Cell<TSP>::RegisterCell(Cell<TSP> *c) {
  std::lock_guard<std::mutex> lock(data_->ht_mtx_);
  data_->ht_[c->key()] = c;
}

template <class TSP>
Region<TSP> Cell<TSP>::CalcRegion(KeyType key, const Region<TSP> &region) {
  if (key == 0) return region;
  
  const int kDim = TSP::Dim;
  
  auto r = region;
  const int kDepth = SFC::GetDepth(key);

  for (int dep = 1; dep <= kDepth; dep++) {
    for (int dim = 0; dim < kDim; dim++) {
      FP center = r.min(dim) + r.width(dim) / 2;
      if (SFC::GetDirOnDepth(key, dim, dep) == 1) {
        r.min(dim) = center;
      } else {
        r.max(dim) = center;
      }
    }
  }
  
  return r;
}

template <class TSP>
Vec<Cell<TSP>::Dim, typename TSP::FP> Cell<TSP>::CalcCenter(KeyType key, const Region<TSP> &region) {
  auto r = CalcRegion(key, region);
  return r.min() + r.width() / 2;
}

/**
 * \brief Generate leaves and associated information from bodies.
 * First sort the bodies according to their SFC keys. Then, Split the space 
 * recursively from the root until all leaves have less than `ncrit` bodies.
 * \param [IN] 
 */
template<class TSP>
void GenerateLeaves(int num_bodies,
                    typename Cell<TSP>::BodyType *b,
                    int ncrit,
                    const Region<TSP> &region,
                    int &max_depth,
                    std::vector<HelperNode<TSP>> &hn,
                    std::vector<typename Cell<TSP>::SFC::KeyType> &leaf_keys,
                    std::vector<index_t> &leaf_nb_local,
                    std::vector<index_t> &leaf_nb_global) {
  // Sort local bodies using SFC  keys
  using HN = HelperNode<TSP>;
  using SFC = typename Cell<TSP>::SFC;
  (void) region; // to avoid unsed parameter warnings.
  (void) b;
  (void) num_bodies;
  
  std::sort(hn.begin(), hn.end(),
            [](const HN &lhs, const HN &rhs) { return lhs.key < rhs.key; });

  int tmp_max_depth;

  // Start from a root cell and refine it recursively until all cells have at most
  leaf_keys.push_back(0);

  // Loop until all leaf cells have at most max_nb_ bodies.
  while(1) {
    leaf_nb_local.clear();
    leaf_nb_global.clear();
    leaf_nb_local.resize(leaf_keys.size(), 0);
    leaf_nb_global.resize(leaf_keys.size(), 0);
    
    for (size_t i = 0; i < leaf_keys.size(); i++) {
      auto _key = [](const HN &hn) { return hn.key; };
      // Count process-local bodies belonging to the cell[i].
      leaf_nb_local[i] = GetBodyRange<SFC, HelperNode<TSP>>(leaf_keys[i],
                                                            hn,
                                                            _key).second;
    }
    
    // Count bodies belonging to the cell[i] globally using MPI_Allreduce(+)
    MPI_Allreduce(leaf_nb_local, leaf_nb_global, MPI_SUM, MPI_COMM_WORLD);

    long max_nb = *std::max_element(leaf_nb_global.begin(), leaf_nb_global.end());
    long total_nb = 0;
    for (auto nb : leaf_nb_global) {
      total_nb += nb;
    }
    
#ifdef TAPAS_DEBUG
    BarrierExec([&] (int rank, int) {
        if (rank == 0) {
          std::cerr << "rank " << std::fixed << std::setw(3) << std::left << rank << "  ";
          std::cerr << "leaf_nb_global.size() = " << leaf_nb_global.size() << ", [";
          for (auto nb : leaf_nb_global) {
            std::cerr << std::fixed << std::setw(3) << nb << " ";
          }
          std::cerr << "]" << std::endl;
          std::cerr << "Total nb = " << total_nb << std::endl;
          std::cerr << "rank " << rank << " " << "max_nb = " << max_nb << " "
                    << "(the limit is " << ncrit << ")" << std::endl;
          std::cerr << std::endl;
        }
      });
#endif
    
    if (max_nb <= ncrit) {    // Finished. all cells have at most max_nb_ bodies.
      break;
    } else {
      // Find cells that have more than max_nb_ bodies and split them.
      leaf_keys = SplitLargeCellsOnce<TSP>(leaf_keys, leaf_nb_global, ncrit, &tmp_max_depth);
    }
  } // end of while(1) loop
  
  tapas::mpi::Allreduce(&tmp_max_depth, &max_depth, 1, MPI_SUM, MPI_COMM_WORLD);
}

/**
 * \brief An functional utility that provides upward reudce of a local tree for internal use.
 *
 * Non-local cells are just ignored.
 * Values of member variables (memvp) of child cells are reduce using function f and
 * assigned into parent's memvp value.
 * If c is a leaf, c.*memvp is set to `init`
 * 
 * \tparam T    Type of target member values.
 * \tparam Funct Type of the reducing function f. It is expected to be T(T, KeType, const CellType*c).
 * \param c     Starting root cell
 * \param memvp A member variable pointer (obtained by &Class::member)
 * \param init  Initial value
 * \param f     A reducing function of type (T, KeyType, const Cell*) -> T. If a child cell is local, 
 *              KeyType and Cell* are both given. If a child is not local, the pointer is nullptr.
 */
template<class TSP, class T, class Funct>
void LocalUpwardReduce(Cell<TSP> *c, T Cell<TSP>::*memvp, T init, Funct f) {
  using KeyType = typename Cell<TSP>::KeyType;
  using SFC = typename Cell<TSP>::SFC;
  
  if (c->IsLeaf()) {
    // If c is a leaf, just assign the value `init` to `memvp` member variable.
    c->*memvp = init;
  } else {
    // c is a non-leaf cell. Reduce children's value and assign it into c->*memvp
    auto &data = c->data();
    KeyType key = c->key();
    auto children_keys = SFC::GetChildren(key);

    T val = init;
    
    for (auto chk : children_keys) {
      if (data.ht_.count(chk) > 0) {
        Cell<TSP> *cc = data.ht_.at(chk);
        LocalUpwardReduce(cc, memvp, init, f);
        val = f(val, chk, const_cast<const Cell<TSP>*>(cc));
      } else {
        val = f(val, chk, nullptr);
      }
    }
    
    c->*memvp = val; // assign reduced value.
  }
}

/**
 * \brief Performs pre-order traversal for local cells.
 * \tparam TSP TSP.
 * \tparam Funct A callback function called for each cell. Expected to be <bool (CellType*)>.
 * \param c A cell to start traversal
 * \param f A callback function. If f returns false for a cell, then its children are not visited.
 */
template<class TSP, class Funct>
void LocalPreOrderTraverse(Cell<TSP> *c, Funct f) {
  using SFC = typename Cell<TSP>::SFC;
  
  TAPAS_ASSERT(c->IsLocal());

  auto &data = c->data();

  bool cont = f(c);
  if (!cont) return;

  if (!c->IsLeaf()) {
    auto child_keys = SFC::GetChildren(c->key());
    for (auto chk : child_keys) {
      if (data.ht_.count(chk) > 0) {
        Cell<TSP> *cc = data.ht_.at(chk);
        LocalPreOrderTraverse(cc, f);
      }
    }
  }
}

/**
 * \brief A reducing function used with LocalUpwardReduce, to check if a cell is a local subtree.
 */
template<class KeyType, class CellType>
bool ReduceLocality(bool b, KeyType, const CellType *c) {
  return b && c != nullptr && c->IsLocalSubtree();
}

template<class TSP>
void FindLocalRoots(typename Cell<TSP>::KeyType key,
                    const typename Cell<TSP>::CellHashTable &ht,
                    typename Cell<TSP>::KeySet   &lroots) {
  using CellType = Cell<TSP>;
  using KeyType = typename CellType::KeyType;
  //using SFC = typename CellType::SFC;

  TAPAS_ASSERT(ht.count(key) == 1);

  CellType *c = ht.at(key);

  LocalUpwardReduce(c, &CellType::is_local_subtree_, true,
                    ReduceLocality<KeyType, CellType>);

  // Create a closure to find all local roots.
  auto local_root_collector = [&lroots] (const CellType *c) -> bool {
    if (c->is_local_subtree_) {
      lroots.insert(c->key());
      return false;
    } else {
      return true;
    }
  };
  
  // Find all local subtree roots.
  LocalPreOrderTraverse(c, local_root_collector);
}

template<class TSP>
void ExchangeGlobalLeafKeys(const typename Cell<TSP>::KeySet &lroots,
                            typename Cell<TSP>::KeySet &gleaves) {
  using KeyType = typename Cell<TSP>::KeyType;

  std::vector<KeyType> gl_keys_send(lroots.begin(), lroots.end()); // global leaf keys
  std::vector<KeyType> gl_keys_recv;
  tapas::mpi::Allgatherv(gl_keys_send, gl_keys_recv, MPI_COMM_WORLD);
  
  gleaves.insert(gl_keys_recv.begin(), gl_keys_recv.end());
}

template<class TSP>
void GrowGlobalTree(const typename Cell<TSP>::KeySet &gleaves,
                    const typename Cell<TSP>::CellHashTable &ht,
                    typename Cell<TSP>::CellHashTable &ht_gtree) {
  using CellType = Cell<TSP>;
  using KeyType = typename CellType::KeyType;
  using SFC = typename CellType::SFC;
  using Data = HotData<TSP, SFC>;
  
  std::shared_ptr<Data> dp = ht.at(0)->data_ptr();

  for (auto && key : gleaves) {
    for (KeyType k = key; k != 0; k = SFC::Parent(k)) {
      if (ht_gtree.count(k) > 0) {
        break; // break to outer loop
      } else if (ht.count(k) > 0) {
        ht_gtree[k] = ht.at(k);
      } else {
        ht_gtree[k] = CellType::CreateRemoteCell(k, 0, dp);
      }
    }
  }
  ht_gtree[0] = ht.at(0);
}

/**
 * \brief Build global tree
 * 1. Traverse recursively from the root and identify global leaves
 * 2. Exchange global leaves using Alltoallv
 * 3. Build global tree locally
 */
template<class TSP>
void BuildGlobalTree(HotData<TSP, typename Cell<TSP>::SFC> &data) {
  // Traverse from root and mark global leaves
  using HT = typename Cell<TSP>::CellHashTable;
  using ST = typename Cell<TSP>::KeySet;
  //using KeyType = typename Cell<TSP>::KeyType;

  HT &ltree = data.ht_;       // hash table for the local tree
  HT &gtree = data.ht_gtree_; // hash table for the global tree
  ST &gleaves = data.gleaves_;
  ST &lroots = data.lroots_;

  gtree.clear();

  FindLocalRoots<TSP>(0, ltree, lroots);

  // Exchange global leaves using Allgatherv
  ExchangeGlobalLeafKeys<TSP>(lroots, gleaves);

  // Glow the global tree locally in each process
  GrowGlobalTree<TSP>(gleaves, data.ht_, data.ht_gtree_);
}


/**
 * @brief Partition the simulation space and build SFC key based octree
 * @tparam TSP Tapas static params
 * @param b Array of particles
 * @param numBodies Length of b (NOT the total number of bodies over all processes)
 * @param r Geometry of the target space
 * @return The root cell of the constructed tree
 * @todo In this function keys are exchanged using alltoall communication, as well as bodies.
 *       In extremely large scale systems, calculating keys locally again after communication
 *       might be faster.
 */
template <class TSP> // TSP : Tapas Static Params
Cell<TSP>*
Partitioner<TSP>::Partition(typename TSP::BT::type *b,
                            index_t num_bodies,
                            const Region<TSP> &reg) {
  using BodyType = typename TSP::BT::type;
  using BodyAttrType = typename TSP::BT_ATTR;
  
  using HN = HelperNode<TSP>;
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;
  
  typedef Cell<TSP> CellType;
  using Data = typename CellType::Data;

  auto data = std::make_shared<Data>();

  int max_depth = 0;

#ifdef TAPAS_MEASURE
  double beg, end;
  beg = MPI_Wtime();
#endif

  MPI_Comm_rank(MPI_COMM_WORLD, &data->mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &data->mpi_size_);
  int mpi_rank = data->mpi_rank_;
  int mpi_size = data->mpi_size_;

  // Calculate the global bouding box by MPI_Allreduce
  data->region_ = ExchangeRegion(reg);

  auto &leaf_nb_global = data->leaf_nb_;
  auto &leaf_keys = data->leaf_keys_;
  std::vector<index_t> leaf_nb_local;
  
  std::vector<HN> hn = CreateInitialNodes<TSP>(b, num_bodies, data->region_);
  
  GenerateLeaves<TSP>(num_bodies,
                      b,
                      max_nb_,
                      data->region_,
                      data->max_depth_,
                      hn,
                      leaf_keys,
                      leaf_nb_local,
                      leaf_nb_global);

  if (data->mpi_rank_ == 0) {
    // debug
    std::cerr << "Max depth = " << max_depth << std::endl;
  }

  // distribute the morton-ordered leaf cells over processes
  // so that each process has roughly equal number of bodies.
  // Split the morton-ordred curve and assign cells to processes
  auto &leaf_owners = data->leaf_owners_;
  leaf_owners = SplitKeysSimple(leaf_nb_global, mpi_size);

#ifdef TAPAS_DEBUG
  {
    Stderr e("leaf_owners");
    assert(leaf_keys.size() == leaf_owners.size());
    for (size_t i = 0; i < leaf_keys.size(); i++) {
      e.out() << SFC::Simplify(leaf_keys[i]) << " "
              << leaf_owners[i] << " "
              << leaf_keys[i] << std::endl;
    }
  }
#endif
  
  data->proc_first_keys_ = ProcHeadKeys<KeyType>(leaf_keys, leaf_owners, mpi_size);

  // Exchange bodies using MPI_Alltoallv
  std::vector<int> send_counts(mpi_size, 0); // number of bodies that this process sends to others
  std::vector<int> recv_counts(mpi_size, 0); // number of bodies that this process receives from others
  
  // count bodies to be sent to process 'proc' in cell ci
  // note that send_count and recv_count are multiplied by sizeof(BodyType) so that
  // BodyType objects will be sent as arrays of MPI_BYTE
  for (size_t ci = 0; ci < leaf_keys.size(); ci++) {
    int proc = leaf_owners[ci];
    send_counts[proc] += leaf_nb_local[ci];
  }

  tapas::mpi::Alltoall(send_counts, recv_counts, 1, MPI_COMM_WORLD);
#if 0
  MPI_Alltoall(send_counts.data(), 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               MPI_COMM_WORLD);
#endif
  
  std::vector<int> send_bytes_bodies(send_counts);
  std::vector<int> send_bytes_keys(send_counts);
  
  std::vector<int> recv_bytes_bodies(recv_counts);
  std::vector<int> recv_bytes_keys(recv_counts);

  for (auto &c : send_bytes_bodies) { c *= sizeof(BodyType); }
  for (auto &c : recv_bytes_bodies) { c *= sizeof(BodyType); }
  for (auto &c : send_bytes_keys) { c *= sizeof(KeyType); }
  for (auto &c : recv_bytes_keys) { c *= sizeof(KeyType); }

  // Exchange particle using MPI_Alltoallv
  // Since Tapas framework does not know the detail of BodyType, we send/recv BodyType data
  // as arrays of MPI_BYTE.

  // Copy bodies and keys to send_buf.
  // hn (array of HelperNode) is already sorted by their SFC keys,
  // which also means they are sorted by their parent process.
  std::vector<BodyType> send_bodies(num_bodies);
  std::vector<KeyType>  send_keys(num_bodies);
  for (size_t hi = 0; hi < hn.size(); hi++) {
    int bi = hn[hi].p_index;
    send_bodies[hi] = b[bi];
    send_keys[hi] = hn[hi].key;
  }

  // Calculate send displacements, which is prefix sum of send_bytes_bodies
  std::vector<int> send_disp_bodies(mpi_size, 0);
  std::vector<int> send_disp_keys(mpi_size, 0);

  for (int p = 0; p < mpi_size; p++) {
    // p : process id (i.e. rank in MPI)
    if (p == 0) {
      send_disp_bodies[p] = 0;
      send_disp_keys[p] = 0;
    } else {
      // find cells that belong to process p.
      const auto beg = leaf_owners.begin();
      const auto end = leaf_owners.end();
      int p_beg_idx = std::lower_bound(beg, end, p-1) - beg; // beg of process p's cells
      int p_end_idx = std::upper_bound(beg, end, p-1) - beg; // end of process p's cells

      // calculate total number of bodies that are sent to process p
      // and the displacement for process p.
      int nbodies = std::accumulate(leaf_nb_local.begin() + p_beg_idx,
                                    leaf_nb_local.begin() + p_end_idx,
                                    0);
      send_disp_bodies[p] = nbodies * sizeof(BodyType) + send_disp_bodies[p-1];
      send_disp_keys[p] = nbodies * sizeof(KeyType) + send_disp_keys[p-1];
    }
  }

  // Prepare local_bodies (array of received body, which will be used in local computation)
  // and local_body_keys (array of keys corresponding to local_bodies)
  // local_bodies will be held by cells and continuously used after this function.
  int num_bodies_recv = sum(recv_counts); // note: recv_count is in bytes.
  auto &local_bodies = data->local_bodies_;
  auto &local_body_keys = data->local_body_keys_;
  auto &local_body_attrs = data->local_body_attrs_;

  local_bodies.resize(num_bodies_recv);
  local_body_keys.resize(num_bodies_recv);
  local_body_attrs.resize(num_bodies_recv);
  
  bzero(reinterpret_cast<void*>(local_body_attrs.data()),
        sizeof(BodyAttrType) * local_body_attrs.size());

  // Calculate recv displacement, which is prefix sum of recv_bytes_bodies and recv_bytes_keys
  std::vector<int> recv_disp_bodies(mpi_size, 0);
  std::vector<int> recv_disp_keys(mpi_size, 0);
  for (size_t pi = 0; pi < recv_counts.size(); pi++) {
    if (pi == 0) {
      recv_disp_bodies[pi] = 0;
      recv_disp_keys[pi] = 0;
    } else {
      recv_disp_bodies[pi] = recv_disp_bodies[pi-1] + recv_bytes_bodies[pi-1];
      recv_disp_keys[pi] = recv_disp_keys[pi-1] + recv_bytes_keys[pi-1];
    }
  }
  
  // Call MPI_Alltoallv() for bodies
  MPI_Alltoallv(send_bodies.data(),  send_bytes_bodies.data(), send_disp_bodies.data(), MPI_BYTE,
                local_bodies.data(), recv_bytes_bodies.data(), recv_disp_bodies.data(), MPI_BYTE,
                MPI_COMM_WORLD);

  // Call MPI_Alltoallv() for body keys
  MPI_Alltoallv(send_keys.data(), send_bytes_keys.data(), send_disp_keys.data(), MPI_BYTE,
                local_body_keys.data(), recv_bytes_keys.data(), recv_disp_keys.data(), MPI_BYTE,
                MPI_COMM_WORLD);

  // Now we have all bodies & keys transferred to their owner processes.
  // Sort the bodies locally using their keys.
  SortByKeys(local_body_keys, local_bodies);

  // Dump local bodies into a file named exch_bodies.dat
  // All processes dump bodies in the file in a coordinated way. init_bodies.dat and
  // exch_bodies.dat must match (if sorted).
#ifdef TAPAS_DEBUG
  BarrierExec([&](int rank, int size) {
      std::stringstream ss;
      ss << "exch_bodies." << size << ".dat";
      bool append_mode = (rank > 0);
      DumpToFile(local_bodies, local_body_keys, ss.str().c_str(), append_mode);
    });
#endif

  // construct a local tree from cells which belong to this process.
  auto leaf_beg = std::lower_bound(std::begin(leaf_owners), std::end(leaf_owners), mpi_rank)
                  - std::begin(leaf_owners);
  auto leaf_end = std::upper_bound(std::begin(leaf_owners), std::end(leaf_owners), mpi_rank)
                  - std::begin(leaf_owners);

  std::vector<Cell<TSP>*> interior_cells;
  
  // Build a local tree in a bottom-up manner.
  for (auto i = leaf_beg; i < leaf_end; i++) {
    KeyType k = leaf_keys[i];
    //KeyType kn = SFC::GetNext(k);

    // Create a leaf cell
    auto *c = Cell<TSP>::CreateLocalCell(k, data);
    // CellType *c = new CellType(k,               // key
    //                            true,            // is_local
    //                            bbeg,            // body index
    //                            bend - bbeg,     // #bodies
    //                            data);
    data->ht_[k] = c;
    assert(c->IsLocal() && c->IsLeaf());

    // Create anscestors of the cell c (in a recursive upward way)
    while(1) {
      k = SFC::Parent(k);
      int dp = SFC::GetDepth(k);

      if (data->ht_.count(k) == 0) {
        index_t bbeg, bend;
        //FindRangeByKey<TSP>(local_body_keys, k, bbeg, bend);
        SFC::FindRangeByKey(leaf_keys, k, bbeg, bend);
        int nb = 0;
        for (auto i = bbeg; i < bend; i++) {
          nb += leaf_nb_global[i];
        }
        
        // Create interior cellls (anscestors)
        // Note:
        // If a cell is non-leaf, then bbeg (body begin index) is not correct.
        // This is because bodies are help only by a process that owns the corresponding leaf cells.
        auto *c = Cell<TSP>::CreateLocalCell(k, data);
        // CellType *c = new CellType(k,     // key
        //                            true,  // is_local
        //                            0, nb, // Read the note above
        //                            data);
        data->ht_[k] = c;
        interior_cells.push_back(c);
        assert(c->IsLocal() && !c->IsLeaf());
      } else {
        break; // if c's parent is found: all of the ancestors have already been created.
      }
      
      if (dp == 0) break; // stop if k is the root cell;
    }
  }
  // we have created all local cells

#ifdef TAPAS_DEBUG
  // Debug
  // Dump all (local) cells to a file
  {
    for (auto& iter : data->ht_) {
      KeyType k = iter.first;
      Cell<TSP> *c = iter.second;
      if (c->IsLocal() && c->key() != 0) {
#ifdef TAPAS_DEBUG
        Stderr e("cells");
        e.out() << SFC::Simplify(k) << " "
                << "d=" << SFC::GetDepth(k) << " "
                << "leaf=" << c->IsLeaf() << " "
                << "owners=" << (c->IsLeaf() ? FindOwnerByKey(leaf_keys, leaf_owners, c->key()) : -1) << " "
                << "nb=" << std::setw(3) << (c->IsLeaf() ? tapas::debug::ToStr(c->nb()) : "N/A") << " "
            //<< "center=[" << c->center() << "] "
            //<< "next_key=" << SFC::Simplify(SFC::GetNext(k)) << " "
            //<< "parent=" << SFC::Simplify(SFC::Parent(k)) << " "
                << "decoded=" << SFC::Decode(k) << " "
                << std::endl;
        if (c->IsLeaf() && c->nb() == 1) {
          e.out() << "Particle ["
                  << c->body(0).x << ", "
                  << c->body(0).y << ", "
                  << c->body(0).z << ", "
                  << c->body(0).w << "]" << std::endl;
        }
#endif
        // Print bodies which belong to Cell c
#if 0
        if (c->IsLeaf()) {
          index_t body_beg, body_end;
          SFC::FindRangeByKey(local_body_keys, k, body_beg, body_end);
          for (int i = body_beg; i < body_end; i++) {
            e.out() << "\t\t\t| "
                    << SFC::Simplify(local_body_keys[i]) << ": "
                    << local_bodies[i].X
                    << std::endl;
          }
        }
#endif
      }
    }
  }
#endif // TAPAS_DEBUG
  
  if (data->ht_[0] == nullptr) {
    // If no leaf is assigned to the process, root node is not generated
    if (data->mpi_rank_ == 0) {
      std::cerr << "There are too few particles compared to the number of processes."
                << std::endl;
    }
    MPI_Finalize();
    exit(-1);
  }

#ifdef TAPAS_MEASURE
  end = MPI_Wtime();
  std::cout << "time Partition " << (end - beg) << std::endl;
#endif

  BuildGlobalTree(*data);

  // return the root cell (root key is always 0)
  return data->ht_[0];
}

} // namespace hot

#ifdef _F
# warning "Tapas function macro _F is already defined. "                \
  "Maybe it is conflicting other libraries or you included incompatible tapas headers."
#endif

/**
 * @brief Constructs a ProductIterator for dual tree traversal of two trees
 */
template <class TSP>
ProductIterator<tapas::iterator::CellIterator<hot::Cell<TSP>>,
                tapas::iterator::CellIterator<hot::Cell<TSP>>>
                                    Product(hot::Cell<TSP> &c1,
                                            hot::Cell<TSP> &c2) {
  TAPAS_LOG_DEBUG() << "Cell-Cell product\n";
  typedef hot::Cell<TSP> CellType;
  typedef CellIterator<CellType> CellIterType;
  return ProductIterator<CellIterType, CellIterType>(
      CellIterType(c1), CellIterType(c2));
}

/**
 * @brief A partitioning plugin class that provides SFC-curve based octree partitioning.
 */
template<int _Dim,
         template<int __Dim, class __KeyType> class _SFC,
         class _KeyType = uint64_t>
struct HOT {
  using SFC = _SFC<_Dim, _KeyType>;
  using KeyType = typename SFC::KeyType;
};

/**
 * @brief Advance decleration of a dummy class to achieve template specialization.
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR,
          class PartitionAlgorithm,
          class Threading>
class Tapas;

#if 0
/**
 * @brief Specialization of Tapas for HOT (Morton HOT) algorithm
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR,
          class Threading>
class Tapas<DIM, FP, BT, BT_ATTR, CELL_ATTR, MortonHOT, Threading> {
  typedef TapasStaticParams<DIM, FP, BT, BT_ATTR, CELL_ATTR, Threading> TSP; // Tapas static params
 public:
  using Region = tapas::Region<TSP>;
  using Cell = hot::Cell<TSP>;
  using SFC = tapas::sfc::Morton<DIM>;
  
  /**
   * @brief Partition and build an octree of the target space.
   * @param b Array of body of BT::type.
   */
  static Cell *Partition(typename BT::type *b,
                         index_t nb, const Region &r,
                         int max_nb) {
    hot::Partitioner<TSP> part(max_nb);
    return part.Partition(b, nb, r);
  }
};
#endif

namespace threading {
class MassiveThreads;
}

/**
 * @brief Specialization of Tapas for HOT (Morton HOT) algorithm
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR, class Threading>
class Tapas<DIM, FP, BT, BT_ATTR, CELL_ATTR, HOT<DIM, tapas::sfc::Morton>, Threading> {
  
  typedef HOT<DIM, tapas::sfc::Morton> MortonHOT;
  
  typedef TapasStaticParams<DIM, FP, BT, BT_ATTR, CELL_ATTR, Threading,
                            typename MortonHOT::SFC> TSP; // Tapas static params
 public:
  using Region = tapas::Region<TSP>;
  using Cell = hot::Cell<TSP>;

  using SFC = typename TSP::SFC;
  
  /**
   * @brief Partition and build an octree of the target space.
   * @param b Array of body of BT::type.
   */
  static Cell *Partition(typename BT::type *b,
                         index_t nb, const Region &r,
                         int max_nb) {
    hot::Partitioner<TSP> part(max_nb);
    return part.Partition(b, nb, r);
  }
};

} // namespace tapas

#ifdef TAPAS_DEBUG
template<class TSP>
std::ostream& operator<<(std::ostream& os, tapas::hot::Cell<TSP> &cell) {
  using CellType = tapas::hot::Cell<TSP>;
  using SFC = typename CellType::SFC;

  os << "Cell: " << "key     = " << cell.key() << std::endl;
  os << "      " << "        = " << SFC::Decode(cell.key()) << std::endl;
  os << "      " << "        = " << SFC::Simplify(cell.key()) << std::endl;
  os << "      " << "IsLeaf  = " << cell.IsLeaf() << std::endl;
  os << "      " << "IsLocal = " << cell.IsLocal() << std::endl;
  if (cell.IsLeaf()) {
    os << "      " << "nb      = " << cell.nb() << std::endl;
  } else {
    os << "      " << "nb      = " << "N/A" << std::endl;
  }
  return os;
}
#endif

#endif // TAPAS_HOT_
