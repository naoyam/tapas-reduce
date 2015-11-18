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

#include "tapas/hot/data.h"
#include "tapas/hot/buildtree.h"
#include "tapas/hot/global_tree.h"
#include "tapas/hot/let.h"
#include "tapas/hot/report.h"

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
  friend class SamplingOctree<TSP, typename TSP::SFC>;
  friend class GlobalTree<TSP>;
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
    c->is_local_ = false; // This cell is 
    c->is_local_subtree_ = false;
    c->nb_ = nb;
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

  //BodyAttrType *body_attrs();
  //const BodyAttrType *body_attrs() const;
  
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

  void Report() const { tapas::hot::Report<Data>(*data_); }
  
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
  inline Cell *Lookup(KeyType k) const;
  CellHashTable *ht() { return ht_; }

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

  void CheckBodyIndex(index_t idx) const;
}; // class Cell

template<class T>
using uset = std::unordered_set<T>;

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
  tapas::mpi::Allreduce(&r.max()[0], &new_max[0], Dim, MPI_MAX, MPI_COMM_WORLD);

  // Exchange min
  tapas::mpi::Allreduce(&r.min()[0], &new_min[0], Dim, MPI_MIN, MPI_COMM_WORLD);

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
  if (c1.key() == 0 && c2.key() == 0) {
    LET<TSP>::Exchange(f, c1);
  }
  
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
inline Cell<TSP> *Cell<TSP>::Lookup(KeyType k) const {
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
void Cell<TSP>::CheckBodyIndex(index_t idx) const {
  (void)idx;
  TAPAS_ASSERT(this->nb() >= 0);
  TAPAS_ASSERT((size_t)idx < this->nb());
  TAPAS_ASSERT(this->IsLeaf() && "body or body attribute access is not allowed for non-leaf cells.");

  if (is_local_) {
    TAPAS_ASSERT(this->bid() + idx < data_->local_bodies_.size());
  } else {
    TAPAS_ASSERT(this->bid() + idx < data_->let_bodies_.size());
  }
}

template <class TSP>
const typename TSP::BT::type &Cell<TSP>::body(index_t idx) const {
  CheckBodyIndex(idx);
  
  if (is_local_) {
    return data_->local_bodies_[this->bid() + idx];
  } else {
    return data_->let_bodies_[this->bid() + idx];
  }
}

template <class TSP>
typename TSP::BT::type &Cell<TSP>::body(index_t idx) {
  return const_cast<typename TSP::BT::type &>(const_cast<const Cell<TSP>*>(this)->body(idx));
}

template <class TSP>
const typename TSP::BT::type &Cell<TSP>::local_body(index_t idx) const {
  TAPAS_ASSERT(this->IsLocal() && "Cell::local_body() can be called only for local cells.");

  TAPAS_ASSERT(idx < data_->local_bodies_.size());

  // TODO is it correct?
  return data_->local_bodies_[this->bid() + idx];
}

template <class TSP>
typename TSP::BT::type &Cell<TSP>::local_body(index_t idx) {
  return const_cast<typename TSP::BT::type &>(const_cast<const Cell<TSP>*>(this)->local_body(idx));
}

// template <class TSP>
// const typename TSP::BT_ATTR *Cell<TSP>::body_attrs() const {
//   CheckBodyIndex(0);
  
//   if (is_local_) {
//     return data_->local_body_attrs_.data() + this->bid();
//   } else {
//     return data_->let_body_attrs_.data() + this->bid();
//   }
// }

// template <class TSP>
// typename TSP::BT_ATTR *Cell<TSP>::body_attrs() {
//   return const_cast<typename TSP::BT_ATTR &>(const_cast<const Cell<TSP>*>(this)->local_attrs());
// }

template <class TSP>
const typename TSP::BT_ATTR &Cell<TSP>::body_attr(index_t idx) const {
  CheckBodyIndex(idx);
  
  if (is_local_) {
    return this->data_->local_body_attrs_[this->bid() + idx];
  } else {
    return this->data_->let_body_attrs_[this->bid() + idx];
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
  static void SelectResponseCells(std::vector<KeyType> &cell_attr_keys, std::vector<int> &attr_src_pids,
                                  std::vector<KeyType> &leaf_keys, std::vector<int> &leaf_src_pids,
                                  const HT& hash) {
    std::set<std::pair<int, KeyType>> res_attr; // keys (and their destinations) of which attributes will be sent as response.
    std::set<std::pair<int, KeyType>> res_body; // keys (and their destinations) of which bodies will be sent as response.

    TAPAS_ASSERT(cell_attr_keys.size() == attr_src_pids.size());
    TAPAS_ASSERT(leaf_keys.size() == leaf_src_pids.size());

    for (size_t i = 0; i < cell_attr_keys.size(); i++) {
      KeyType k = cell_attr_keys[i];
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
        
    for (size_t i = 0; i < leaf_keys.size(); i++) {
      KeyType k = leaf_keys[i];
      int src_pid = leaf_src_pids[i];

      while(hash.count(k) == 0) {
        k = SFC::Parent(k);
      }

      TAPAS_ASSERT(k != 0); // the same reason above
      TAPAS_ASSERT(hash.count(k) > 0);
      TAPAS_ASSERT(hash.at(k)->IsLeaf());

      res_body.insert(std::make_pair(src_pid, k));
    }

#if TAPAS_DEBUG
    BarrierExec([&res_attr, &res_body](int rank, int) {
        std::cerr << "Rank " << rank << " SelectResponseCells: keys_attr.size() = " << res_attr.size() << std::endl;
        std::cerr << "Rank " << rank << " SelectResponseCells: keys.body.size() = " << res_body.size() << std::endl;
      });
#endif

    // Set values to the vectors
    cell_attr_keys.resize(res_attr.size());
    attr_src_pids.resize(res_attr.size());

    int idx = 0;
    for (auto & iter : res_attr) {
      attr_src_pids[idx] = iter.first;
      cell_attr_keys[idx] = iter.second;
      idx++;
    }

    leaf_keys.resize(res_body.size());
    leaf_src_pids.resize(res_body.size());

    idx = 0;
    for (auto & iter : res_body) {
      leaf_src_pids[idx] = iter.first;
      leaf_keys[idx] = iter.second;

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

  static void KeysToBodies(const std::vector<KeyType> &keys,
                           std::vector<index_t> &nb,
                           std::vector<BodyType> &bodies,
                           const HT& hash) {
    nb.resize(keys.size());
    bodies.clear();

    // In BH, each leaf has 0 or 1 body (while every cell has attribute)
    for (size_t i = 0; i < keys.size(); i++) {
      KeyType k = keys[i];
      auto *c = hash.at(k);
      nb[i] = c->IsLeaf() ? c->nb() : 0;

      for (size_t bi = 0; bi < nb[i]; bi++) {
        bodies.push_back(c->body(bi));
      }
    }
  }
  
}; // class Partitioner

/**
 * @brief Overloaded version of Partitioner::Partition
 */
template <class TSP>
Cell<TSP>*
Partitioner<TSP>::Partition(std::vector<typename TSP::BT::type> &b, const Region<TSP> &r) {
    return Partitioner<TSP>::Partition(b.data(), b.size(), r);
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
  using SFC = typename TSP::SFC;
  using CellType = Cell<TSP>;
  using Data = typename CellType::Data;

  auto data = std::make_shared<Data>();

  MPI_Comm_rank(MPI_COMM_WORLD, &data->mpi_rank_);
  MPI_Comm_size(MPI_COMM_WORLD, &data->mpi_size_);

  SamplingOctree<TSP, SFC> stree(b, num_bodies, reg, data, max_nb_);
  stree.Build();
    
#ifdef TAPAS_USE_VECTORMAP
  using BodyType = typename TSP::BT::type;
  using BodyAttrType = typename TSP::BT_ATTR;
  
  /* (No templates allowed.) */
  typedef typename TSP::Vectormap:: template um_allocator<BodyType>
    body_vector_allocator;
  typedef typename TSP::Vectormap:: template um_allocator<BodyAttrType>
    attr_vector_allocator;
#endif /*TAPAS_USE_VECTORMAP*/

  GlobalTree<TSP>::Build(*data);

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
          class Threading
#ifdef TAPAS_USE_VECTORMAP
          , class Vectormap
#endif /*TAPAS_USE_VECTORMAP*/
          >
class Tapas;

#if 0
/**
 * @brief Specialization of Tapas for HOT (Morton HOT) algorithm
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR,
          class Threading>
class Tapas<DIM, FP, BT, BT_ATTR, CELL_ATTR, MortonHOT, Threading
#ifdef TAPAS_USE_VECTORMAP
#  ifdef __CUDACC__
            , tapas::Vectormap_CUDA_Packed<DIM, FP, BT, BT_ATTR>
#  else
            , tapas::Vectormap_CPU<DIM, FP, BT, BT_ATTR>
#  endif /*__CUDACC__*/
#endif /*TAPAS_USE_VECTORMAP*/
            > {
  typedef TapasStaticParams<DIM, FP, BT, BT_ATTR, CELL_ATTR, Threading
#ifdef TAPAS_USE_VECTORMAP
#  ifdef __CUDACC__
                            , tapas::Vectormap_CUDA_Packed<DIM, FP, BT, BT_ATTR>
#  else
                            , tapas::Vectormap_CPU<DIM, FP, BT, BT_ATTR>
#  endif /*__CUDACC__*/
#endif /*TAPAS_USE_VECTORMAP*/
                            > TSP; // Tapas static params
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
          class BT_ATTR, class CELL_ATTR, class Threading
#ifdef TAPAS_USE_VECTORMAP
          , class Vectormap
#endif /*TAPAS_USE_VECTORMAP*/
>
class Tapas<DIM, FP, BT, BT_ATTR, CELL_ATTR, HOT<DIM, tapas::sfc::Morton>, Threading
#ifdef TAPAS_USE_VECTORMAP
            , Vectormap
#endif /*TAPAS_USE_VECTORMAP*/
> {
  
  typedef HOT<DIM, tapas::sfc::Morton> MortonHOT;
  
  typedef TapasStaticParams<DIM, FP, BT, BT_ATTR, CELL_ATTR, Threading,
                            typename MortonHOT::SFC
#ifdef TAPAS_USE_VECTORMAP
                            , Vectormap
#endif /*TAPAS_USE_VECTORMAP*/
                            > TSP; // Tapas static params
 public:
  using Region = tapas::Region<TSP>;
  using Cell = hot::Cell<TSP>;
  using BodyIterator = typename Cell::BodyIterator;

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

#ifdef AUTO_LET_SLOW
extern volatile double dummy_value;
#endif

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
