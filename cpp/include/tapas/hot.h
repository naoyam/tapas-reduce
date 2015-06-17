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

#include <unistd.h>
#include <mpi.h>

#include "tapas/cell.h"
#include "tapas/bitarith.h"
#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/iterator.h"
#include "tapas/sfc_morton.h"
#include "tapas/threading/default.h"

#define DEBUG_SENDRECV

namespace tapas {

/**
 * @brief Provides MPI-based distributed SFC-based octree partitioning
 */
namespace hot {

// fwd decl
template<class TSP> class Cell;

/**
 * \brief Struct to hold shared data among Cells
 * Never accessed by users directly. Only held by Cells using shared_ptr.
 */
template<class TSP, class SFC>
struct HotData {
  using KeyType = typename SFC::KeyType;
  using CellType = Cell<TSP>;
  using CellHashTable = typename std::unordered_map<KeyType, CellType*>;
  using BodyType = typename TSP::BT::type;
  using BodyAttrType = typename TSP::BT_ATTR;
  
  CellHashTable ht_;
  std::mutex ht_mtx_;  //!< mutex to protect ht_
  Region<TSP> region_; //!< global bouding box

  int mpi_rank;
  int mpi_size;

  std::vector<KeyType> leaf_keys_; //!< SFC keys of (all) leaves
  std::vector<index_t> leaf_nb_;   //!< Number of bodies in each cell
  std::vector<int>     leaf_owners_; //!< Owner process of leaf[i]
  std::vector<BodyType> local_bodies_; //!< Bodies that belong to the local process
  std::vector<KeyType>  local_body_keys_; //!< SFC keys of local bodies
  std::vector<BodyAttrType> local_body_attrs_; //!< Local body attributes

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
 * @brief Sort values using permutations (assuming T1 is an integral type), 
 *        so that vals[i] should be at perms[i]-th position in the resulting vector.
 *        Both of keys and vals are sorted.
 *
 * @param perms Permutations, where vals[i] should be at perms[i]-th in the resulting vector.
 * @param vals Values to be sorted.
 *
 * FIXME: the doc comment above is stale.
 */
template<class T1, class T2>
void SortByPermutations(std::vector<T1> &keys, std::vector<T2> &vals) {
  assert(keys.size() == vals.size());

  auto len = keys.size();

  std::vector<size_t> perm(len);

  for (size_t i = 0; i < len; i++) {
    perm[i] = i;
  }

  std::sort(std::begin(perm), std::end(perm),
            [&keys](size_t a, size_t b) { return keys[a] < keys[b]; });

  std::vector<T1> keys2(len); // sorted keys
  std::vector<T2> vals2(len); // sorted vals
  for (size_t i = 0; i < len; i++) {
    size_t idx = perm[i];
    vals2[i] = vals[idx];
    keys2[i] = keys[idx];
  }
  vals = vals2;
  keys = keys2;
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

// Debug helper function
template<class F>
void BarrierExec(F func) {
  int size, rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Barrier(MPI_COMM_WORLD);
  for (int i = 0; i < size; i++) {
    if (rank == i) {
      func(rank, size);
    }
    usleep(1000000);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

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


template <class TSP>
class Partitioner;

template <class TSP> // TapasStaticParams
class Cell: public tapas::BasicCell<TSP> {
  friend class Partitioner<TSP>;
  friend class BodyIterator<Cell>;

  //========================================================
  // Typedefs 
  //========================================================
 public: // public type usings
  static const constexpr int Dim = TSP::Dim;
  typedef typename TSP::SFC SFC;
  typedef typename SFC::KeyType KeyType;
  
  typedef std::unordered_map<KeyType, Cell*> CellHashTable;
  typedef typename TSP::ATTR attr_type;
  typedef typename TSP::BT::type BodyType;
  typedef typename TSP::BT_ATTR BodyAttrType;
  typedef BodyIterator<Cell> BodyIter;
  typedef typename TSP::Threading Threading;

  using FP = typename TSP::FP;

 private: // private type usings
  using Data = HotData<TSP, SFC>;
  
 public:
  
  template<class T>
  using VecPtr = std::shared_ptr<std::vector<T>>;

  //========================================================
  // Member functions
  //========================================================

  /**
   * @brief Constructor of Cell class
   * @param key Key of the cell
   * @param is_local Whether the cell is a local cell
   * @param is_leaf  Whether the cell is a leaf cell
   * @param body_beg If is_leaf is true, starting index of the range of 
   *                 local_bodies and local_body_keys that the cell owns. 
   *                 Otherwise, 0
   * @param body_num If is_leaf is true, the number of bodies the cell owns.
   */
  Cell(KeyType key,
       bool is_local,
       index_t body_beg, index_t body_num,
       std::shared_ptr<Data> data) :
      tapas::BasicCell<TSP>(CalcRegion(key, data->region_), body_beg, body_num),
      key_(key), is_local_(is_local), is_dummy_(false), data_(data),
      nb_(data->local_bodies_.size())
  {
    // Check if I'm a leaf
    is_leaf_ = find(data_->leaf_keys_.begin(),
                    data_->leaf_keys_.end(), key) != data_->leaf_keys_.end();

    // Count number of bodies of this cell (including cells that are indirectly owned bodies)
    if (body_num == 0) {
      index_t beg, end;
      SFC::GetDescendantRange(key, data_->leaf_keys_.begin(), data_->leaf_keys_.end(), beg, end);

      // [beg, end) is the range of cells that belong to the cell
      int nb = 0;
      for (auto i=beg; i < end; i++) {
        nb += data_->leaf_nb_[i];
      }
      this->nb_ = nb;
    }
  }

  // Create a dummy root node
  Cell(std::shared_ptr<Data> data) : tapas::BasicCell<TSP>(data->region_, 0, 0),
      key_(0), is_local_(false), is_leaf_(false), is_dummy_(true), data_(data)
  {
  }
  
  KeyType key() const { return key_; }
  
  bool operator==(const Cell &c) const;
  template <class T>
  bool operator==(const T &) const { return false; }
  bool IsRoot() const;

  static void Map(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f);
  static void Map(Cell<TSP> &c1, Cell<TSP> &c2,
                  std::function<void(Cell<TSP>&, Cell<TSP>&)> f);
  
  static void Map(BodyIter &b1, BodyIter &b2,
                  std::function<void(BodyIter&, BodyIter&)> f) {
    f(b1, b2);
  }

  static void Map(BodyIter &b1,
                  std::function<void(BodyIter)> f) {
    f(b1);
  }

  /**
   * @brief Returns if the cell is a leaf cell
   */
  bool IsLeaf() const;

  /**
   * @brief Returns if the cell is local.
   */
  bool IsLocal() const;

  /**
   * @brief Returns if the cells is dummy, which means that the local process is not assigned any cell, 
   *        thus Map function just skip it.
   */
  bool IsDummy() const { return is_dummy_; }

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
  
#ifdef DEPRECATED
    typename TSP::BT::type &particle(index_t idx) const {
        return body(idx);
    }
#endif

  // Accessor functions to bodies & body attributes
  BodyType &body(index_t idx);
  const BodyType &body(index_t idx) const;
  
  BodyAttrType &body_attr(index_t idx);
  const BodyAttrType &body_attr(index_t idx) const;
  
  BodyAttrType *body_attrs();
  const BodyAttrType *body_attrs() const;
  
  BodyIterator<Cell> bodies();

  int nbodies() const { return nb_; }

#ifdef DEPRECATED
    typename TSP::BT_ATTR *particle_attrs() const {
        return body_attrs();
    }
#endif
  SubCellIterator<Cell> subcells();

  const Region<TSP> &region() const { return data_->region_; }

 protected:
  // utility/accessor functions
  Cell *Lookup(KeyType k) const;
  CellHashTable *ht() { return ht_; }
  virtual void make_pure_virtual() const {}
  void RegisterCell(Cell<TSP> *c);

  static Region<TSP> CalcRegion(KeyType, const Region<TSP>& r);
  //========================================================
  // member variables
  //========================================================
 protected:
  KeyType key_; //!< Key of the cell
  bool is_local_;
  bool is_leaf_;
  bool is_dummy_; //!< A dummy cell is returned from Partition if no leaf cell is assigned to the process.
  std::shared_ptr<Data> data_;

  int nb_; //!< number of bodies in the local process (not bodies under this cell).
  
  std::shared_ptr<CellHashTable> ht_; //!< Hash table of KeyType -> Cell*
  std::shared_ptr<std::mutex>    ht_mtx_; //!< mutex to manipulate ht_
}; // class Cell

template<class T>
using uset = std::unordered_set<T>;

// Copied from bh.cc
template<typename FP, typename T>
static FP distR2(const T &p, const T &q) {
  FP dx = q.x - p.x;
  FP dy = q.y - p.y;
  FP dz = q.z - p.z;
  return dx * dx + dy * dy + dz * dz;
}

#ifdef TAPAS_BH

template<class TSP, class SetType>
void TraverseLET(typename Cell<TSP>::BodyType &p,
                 Cell<TSP> &cell,
                 SetType &list_geo, SetType &list_attr, SetType &list_body) {
  using KeyType = typename Cell<TSP>::KeyType;
  using FP = typename TSP::FP;
  
  // Maximum depth of the tree.
  // For now, we use the theoretical maximum depth of the tree from the space filling curves,
  // but we can use the actual maximum depth obtained by using MPI_Allreduce.
  const constexpr int max_depth = Cell<TSP>::SFC::MAX_DEPTH;

  if (cell.depth() >= max_depth) {
    return;
  }
  
  KeyType k = cell.key();
  Region<TSP> r = cell.region(); // bounding box of the whole domain
  auto child_keys = Cell<TSP>::SFC::GetChildren(k);

  auto comp = [&](KeyType k1, KeyType k2) {
    // auto c1 = SFC::GetCenter<Vec>(k1, r.min(), r.max());
    // auto c2 = SFC::GetCenter<Vec>(k2, r.min(), r.max());

    // FP d1 = (p.x - c1[0]) * (p.x - c1[0]) +
    //         (p.y - c1[1]) * (p.y - c1[1]) +
    //         (p.z - c1[2]) * (p.z - c1[2]);
    
    // FP d2 = (p.x - c2[0]) * (p.x - c2[0]) +
    //         (p.y - c2[1]) * (p.y - c2[1]) +
    //         (p.z - c2[2]) * (p.z - c2[2]);
    // return d1 < d2;
    return 0;
  };

  // Sort children according to their distance from p
  std::sort(std::begin(child_keys), std::end(child_keys), comp);

  if (cell.IsLeaf()) {
  } else {
    for (auto &chk : child_keys) {
      //auto ctr = Cell<TSP>::SFC::GetCenter(k, r.min(), r.max());
      
      //FP d2 = distR2<FP>(c2.attr(), p1);
    }
  }
         
  return;
}

template<class TSP>
void ExchangeLET(Cell<TSP> &root) {
  using BodyType = typename Cell<TSP>::BodyType;
  using KeyType = typename Cell<TSP>::KeyType;
  using KeySet = typename Cell<TSP>::SFC::KeySet;
  
  KeySet list_geo;  // cells of which geometry (coordinates) are to be transfered.
  KeySet list_attr; // cells of which attributes are to be transfered.
  KeySet list_body; // cells of which bodies are to be transfered

  list_attr.insert(root.key());
  
  for (int bi = 0; bi < root.nbodies(); bi++) {
    BodyType &b = root.body(bi);
    TraverseLET<TSP, KeySet>(b, root, list_geo, list_attr, list_body);
  }
}

#endif // TAPAS_BH

// MPI-related utilities and wrappers
// TODO: wrap them as a pluggable policy/traits class
template<class T> struct MPI_DatatypeTraits {};

#define DEF_MPI_DATATYPE(__ctype, __mpitype) \
  template<> struct MPI_DatatypeTraits<__ctype>  { static MPI_Datatype type() { return __mpitype; } };

DEF_MPI_DATATYPE(int,    MPI_INT);
DEF_MPI_DATATYPE(long,   MPI_LONG);
DEF_MPI_DATATYPE(float,  MPI_FLOAT);
DEF_MPI_DATATYPE(double, MPI_DOUBLE);

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
 * @brief 1-parameter Map function for 
 */
template <class TSP>
void Cell<TSP>::Map(Cell<TSP> &cell, std::function<void(Cell<TSP>&)> f) {
#ifdef TAPAS_BH
  //ExchangeLET<TSP>(cell);
#endif
  
  f(cell);
}

/**
 * 2-parameter Map function over cells (apply user function to products of cells)
 */
template<class TSP>
void Cell<TSP>::Map(Cell<TSP> &c1, Cell<TSP> &c2,
                    std::function<void(Cell<TSP>&, Cell<TSP>&)> f) {
  f(c1, c2);
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
bool Cell<TSP>::IsLeaf() const {
    return is_leaf_;
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
    std::lock_guard<std::mutex> lock(*ht_mtx_);
    c = Lookup(child_key);
    if (c == nullptr) {
      // leaf if if key is contained in leaf_keys_

      c = new Cell(child_key, false, 0, 0, data_);
      data_->ht_[child_key] = c;
    }
  }
  
  return *c;
}

template <class TSP>
Cell<TSP> *Cell<TSP>::Lookup(KeyType k) const {
  auto &ht = data_->ht_;
  auto i = ht.find(k);
  if (i != ht.end()) {
    return i->second;
  } else {
    return nullptr;
  }
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
typename TSP::BT::type &Cell<TSP>::body(index_t idx) {
  assert(idx < this->nb());
  return data_->local_bodies_[this->bid() + idx];
}

template <class TSP>
const typename TSP::BT::type &Cell<TSP>::body(index_t idx) const {
  assert(idx < this->nb());
  return data_->local_bodies_[this->bid() + idx];
}

template <class TSP>
typename TSP::BT_ATTR *Cell<TSP>::body_attrs() {
  return data_->local_body_attrs_.data();
}

template <class TSP>
const typename TSP::BT_ATTR *Cell<TSP>::body_attrs() const {
  return data_->local_body_attrs_.data();
}

template <class TSP>
typename TSP::BT_ATTR &Cell<TSP>::body_attr(index_t idx) {
  assert(idx < this->nb());
  return data_->local_body_attrs_[this->bid() + idx];
}

template <class TSP>
const typename TSP::BT_ATTR &Cell<TSP>::body_attr(index_t idx) const {
  TAPAS_ASSERT(idx < this->nb());
  return data_->local_body_attrs_[this->bid() + idx];
}

template <class TSP>
SubCellIterator<Cell<TSP>> Cell<TSP>::subcells() {
  return SubCellIterator<Cell>(*this);
}

template <class TSP>
BodyIterator<Cell<TSP>> Cell<TSP>::bodies() {
  return BodyIterator<Cell<TSP> >(*this);
}

template <class TSP> // Tapas static params
class Partitioner {
 private:
  const int max_nb_;
  using BodyType = typename TSP::BT::type;
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;

  public:
    Partitioner(unsigned max_nb): max_nb_(max_nb) {}

    Cell<TSP> *Partition(typename TSP::BT::type *b, index_t nb,
                         const Region<TSP> &r);
    Cell<TSP> *Partition(std::vector<typename TSP::BT::type> &b,
                         const Region<TSP> &r);
  private:
    void Refine(Cell<TSP> *c,
                const std::vector<HelperNode<TSP>> &hn,
                const BodyType *b,
                int cur_depth,
                KeyType cur_key) const;
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

#if 0
/**
 * @brief Returns a std::vector of parent's children
 */
template<class TSP>
std::vector<typename TSP::SFC::KeyType> GetChildren(typename TSP::SFC::KeyType parent) {
  using SFC = typename TSP::SFC;
  using KeyType = typename TSP::KeyType;
  
  std::vector<KeyType> ret;
  
  KeyType child_key = SFC::FirstChild(parent);
  for (int child_idx = 0; child_idx < (1 << TSP::Dim); child_idx++) {
    ret.push_back(child_key);
    child_key = SFC::GetNext(child_key);
  }

  return ret;
}
#endif

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
                    int max_nb) {
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;
  
  std::vector<KeyType> ret; // new

  for (size_t i = 0; i < cell_keys.size(); i++) {
    if (nb[i] <= max_nb) {
      // This cell does not need to be split.
      ret.push_back(cell_keys[i]);
    } else {
      // Create 2^DIM children (8 in 3-dim)
      auto children = SFC::GetChildren(cell_keys[i]);
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
  const int kDim = TSP::Dim;
  const int kMask = (1 << TSP::Dim) - 1;
  
  Stderr err("center");

  auto r = region;
  int depth = SFC::GetDepth(key);
  KeyType key_body = SFC::RemoveDepth(key);
  
  err.out() << SFC::Simplify(key) << " "
            << depth << " "
            << r << " --> ";

  for (int d = 0; d < depth; d++) {
    int direction = (key_body >> (kDim * (TSP::SFC::MAX_DEPTH - d - 1))) & kMask;
    r = r.PartitionBSP(direction);
    key >>= TSP::Dim;
    //err.out() << d << ":" << direction << " " << r << " ";
  }
  
  err.out() << r << std::endl;
  
  return r;
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
  
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;
  
  typedef Cell<TSP> CellType;
  typedef HelperNode<TSP> HN;
  using Data = typename CellType::Data;

  auto data = std::make_shared<Data>();

  MPI_Comm_rank(MPI_COMM_WORLD, &data->mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &data->mpi_size);
  int mpi_rank = data->mpi_rank;
  int mpi_size = data->mpi_size;

  // Calculate the global bouding box by MPI_Allreduce
  data->region_ = ExchangeRegion(reg);

  // Sort local bodies using SFC  keys
  std::vector<HN> hn = CreateInitialNodes<TSP>(b, num_bodies, data->region_);
  std::sort(hn.begin(), hn.end(),
            [](const HN &lhs, const HN &rhs) { return lhs.key < rhs.key; });

  BarrierExec([&](int rank, int size) {
      std::vector<BodyType> bodies;
      std::vector<KeyType> keys;

      for(const auto& n : hn) {
        keys.push_back(n.key);
        bodies.push_back(b[n.p_index]);
      }
      bool append_mode = (rank > 0); // Previously existing file is truncated by rank 0
      DumpToFile(bodies, keys, "init_bodies.dat", append_mode);
    });

  // shortcuts to HotData members:
  auto &leaf_keys = data->leaf_keys_;     // Keys of leaf cells (global)
  auto &leaf_nb_global = data->leaf_nb_;  // Number of local bodies (global)

  // Number of local bodies (Allreduce()ed to leaf_nb_global)
  std::vector<index_t> leaf_nb_local;

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
    
#if 0
    BarrierExec([&] (int rank, int size) {
        std::cerr << "rank " << std::fixed << std::setw(3) << std::left << rank << "  ";
        std::cerr << "hn.size() = " << hn.size() << ", ";
        for (auto n : hn) {
          std::cerr << std::fixed << std::setw(6) << SFC::RemoveDepth(n.key) << " ";
        }
        std::cerr << std::endl;

        std::cerr << "rank " << std::fixed << std::setw(3) << std::left << rank << "  ";
        std::cerr << "leaf_nb_local.size() = " << leaf_nb_local.size() << ", ";
        for (auto nb : leaf_nb_local) {
          std::cerr << std::fixed << std::setw(3) << nb << " ";
        }
        std::cerr << std::endl;
      });
#endif

    // Count bodies belonging to the cell[i] globally using MPI_Allreduce(+)
    MPI_Allreduce(leaf_nb_local, leaf_nb_global, MPI_SUM, MPI_COMM_WORLD);

    long max_nb = *std::max_element(leaf_nb_global.begin(), leaf_nb_global.end());
    
    if (max_nb <= max_nb_) {    // Finished. all cells have at most max_nb_ bodies.
      break;
    } else {
      // Find cells that have more than max_nb_ bodies and split them.
      leaf_keys = SplitLargeCellsOnce<TSP>(leaf_keys, leaf_nb_global, max_nb_);
    }
  } // end of while(1) loop

  // distribute the morton-ordered leaf cells over processes
  // so that each process has roughly equal number of bodies.
  // Split the morton-ordred curve and assign cells to processes
  auto &leaf_owners = data->leaf_owners_;
  leaf_owners = SplitKeysSimple(leaf_nb_global, mpi_size);

  std::vector<KeyType> proc_head_keys = ProcHeadKeys<KeyType>(leaf_keys,
                                                              leaf_owners,
                                                              mpi_size);

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

  MPI_Alltoall(send_counts.data(), 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               MPI_COMM_WORLD);

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
  SortByPermutations(local_body_keys, local_bodies);

  // Dump local bodies into a file named exch_bodies.dat
  // All processes dump bodies in the file in a coordinated way. init_bodies.dat and
  // exch_bodies.dat must match (if sorted).
  BarrierExec([&](int rank, int size) {
      std::stringstream ss;
      ss << "exch_bodies." << size << ".dat";
      bool append_mode = (rank > 0);
      DumpToFile(local_bodies, local_body_keys, ss.str().c_str(), append_mode);
    });

  // construct a local tree from cells which belong to this process.
  auto leaf_beg = std::lower_bound(std::begin(leaf_owners), std::end(leaf_owners), mpi_rank)
                  - std::begin(leaf_owners);
  auto leaf_end = std::upper_bound(std::begin(leaf_owners), std::end(leaf_owners), mpi_rank)
                  - std::begin(leaf_owners);

  std::vector<Cell<TSP>*> interior_cells;
  Stderr e("partition");

  // Build a local tree in a bottom-up manner.
  for (auto i = leaf_beg; i < leaf_end; i++) {
    KeyType k = leaf_keys[i];
    //KeyType kn = SFC::GetNext(k);

    // Find bodies owned by the Cell whose key is k.
    index_t bbeg, bend;
    SFC::FindRangeByKey(local_body_keys, k, bbeg, bend);
    
    // Create a leaf cell
    CellType *c = new CellType(k,               // key
                               true,            // is_local
                               bbeg,            // body index
                               bend - bbeg,     // #bodies
                               data);
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
        CellType *c = new CellType(k,     // key
                                   true,  // is_local
                                   0, nb, // Read the note above
                                   data);
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
    Stderr e("cells");
    for (auto&& iter : data->ht_) {
      KeyType k = iter.first;
      Cell<TSP> *c = iter.second;
      if (c->IsLocal() && c->key() != 0) {
        e.out() << SFC::Simplify(k) << " "
                << "d=" << SFC::GetDepth(k) << " "
                << "leaf=" << c->IsLeaf() << " "
                << "nb=" << std::setw(3) << c->nb() << " "
                << "center=[" << c->center() << "] "
                << "next_key=" << SFC::Simplify(SFC::GetNext(k)) << " "
                << "parent=" << SFC::Simplify(SFC::Parent(k)) << " "
                << std::endl;
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
    // 0 : root key 
    // Create a dummy cell if ht[0] is nullptr, which means the local process
    // was assigned no leaf cell (but need to return a root cell).
    data->ht_[0] = new CellType(data);
  }

#ifdef TAPAS_DEBUG
  // Dump all cells in DOT (graphviz) format
  // Only rank 0 process works on this.
  {
    int rank = 0;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0) {
      // Space filling curve key -> owners
      auto owners = std::unordered_map<KeyType, std::unordered_set<int>>();
      auto isleaf = std::unordered_map<KeyType, bool>();
      for (size_t i = 0; i < leaf_owners.size(); i++) {
        KeyType k = leaf_keys[i];
        int owner = leaf_owners[i];
        
        owners[k].insert(owner);
        isleaf[k] = true;
        
        do {
          k = SFC::Parent(k);
          owners[k].insert(owner);
          isleaf[k] = false;
        } while(k != 0);
      }

      std::ofstream ofs("tree_tapas.dot");
      ofs << "digraph tapas_tree {" << std::endl;
      ofs << "graph [rankdir=LR];" << std::endl;

      for (auto iter = owners.begin(); iter != owners.end(); iter++) {
        auto k = iter->first;
        auto &v = iter->second; // Owners

        // Dump cell info
        std::stringstream level;
        level << "[" << SFC::GetDepth(k) << "]";

        std::stringstream owned_by;
        owned_by << "(owned by ";
        for (auto &p : v) owned_by << p << " ";
        owned_by << ")";

        ofs << "cell_" << k << " [label=\"" << SFC::Simplify(k) << " " << level.str() << " " << owned_by.str() << "\"";
        if (isleaf[k]) {
          ofs << ", shape=box";
        }
        ofs << "];" << std::endl;

        // Print link to parent
        if (k != 0) {
          ofs << "cell_" << k << " -> " << "cell_" << SFC::Parent(k) << ";" << std::endl;
        }
      }
      ofs << "}" << std::endl;
    } // if rank 0
  }
#endif
  
  // return the root cell (root key is always 0)
  return data->ht_[0];
}

template <class TSP>
void Partitioner<TSP>::Refine(Cell<TSP> *c,
                              const std::vector<HelperNode<TSP>> &hn,
                              const typename TSP::BT::type *b,
                              int cur_depth,
                              typename TSP::SFC::KeyType cur_key) const {
    const constexpr int Dim = TSP::Dim;
    using SFC = typename TSP::SFC;
    using KeyType = typename SFC::KeyType;
    //using KeyPair = typename SFC::KeyPair;
    //using FP  = typename TSP::FP;
    //using BT  = typename TSP::BT;

    TAPAS_LOG_INFO() << "Current depth: " << cur_depth << std::endl;
    if (c->nb() <= max_nb_) {
        TAPAS_LOG_INFO() << "Small enough cell" << std::endl;
        return;
    }
    if (cur_depth >= TSP::SFC::MAX_DEPTH) {
        TAPAS_LOG_INFO() << "Reached maximum depth" << std::endl;
        return;
    }
    typename SFC::KeyType child_key = SFC::FirstChild(cur_key);
    index_t cur_offset = c->bid();
    index_t cur_len = c->nb();
    
    auto getkey = [](const HelperNode<TSP> &hn) { return hn.key; };

    for (int i = 0; i < (1 << Dim); ++i) {
        TAPAS_LOG_DEBUG() << "Child key: " << child_key << std::endl;
        
        // std::pair<KeyType, KeyType>
        auto kp = GetBodyRange<KeyType, HelperNode<TSP>>(child_key,
                                                         hn.begin() + cur_offset,
                                                         hn.begin() + cur_offset + cur_len,
                                                         getkey);
        index_t child_bn = kp.second;
        TAPAS_LOG_DEBUG() << "Range: offset: " << cur_offset << ", length: "
                          << child_bn << "\n";
        auto child_r = c->region().PartitionBSP(i);
        auto *child_cell = new Cell<TSP>(
            child_r, cur_offset, child_bn, child_key, c->ht(),
            c->bodies_, c->body_attrs_);
        c->ht()->insert(std::make_pair(child_key, child_cell));
#if 0
#ifdef TAPAS_DEBUG
        TAPAS_LOG_DEBUG() << "Particles: \n";
        tapas::debug::PrintBodies<Dim, FP, BT>(b+cur_offset, child_bn, std::cerr);
#endif
#endif
        Refine(child_cell, hn, b, cur_depth+1, child_key);
        child_key = SFC::GetNext(child_key);
        cur_offset = cur_offset + child_bn;
        cur_len = cur_len - child_bn;
    }
    c->is_leaf_ = false;
}

} // namespace hot

template <class TSP, class T2>
ProductIterator<CellIterator<hot::Cell<TSP>>, T2>
Product(hot::Cell<TSP> &c, T2 t2) {
    TAPAS_LOG_DEBUG() << "Cell-X product\n";
    typedef hot::Cell<TSP> CellType;
    typedef CellIterator<CellType> CellIterType;
    return ProductIterator<CellIterType, T2>(CellIterType(c), t2);
}

template <class T1, class TSP>
ProductIterator<T1, CellIterator<hot::Cell<TSP>>>
                         Product(T1 t1, hot::Cell<TSP> &c) {
    TAPAS_LOG_DEBUG() << "X-Cell product\n";
    typedef hot::Cell<TSP> CellType;
    typedef CellIterator<CellType> CellIterType;
    return ProductIterator<T1, CellIterType>(t1, CellIterType(c));
}

/**
 * @brief Constructs a ProductIterator for dual tree traversal of two trees
 */
template <class TSP>
ProductIterator<CellIterator<hot::Cell<TSP>>,
                CellIterator<hot::Cell<TSP>>>
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
 * @brief Find owner process from a head-key list.
 * The argument head_list contains SFC keys that are the first keys of processes.
 * head_list[P] is the first SFC key belonging to process P.
 */
template<class TSP>
int FindOwnerProcess(const std::vector<typename TSP::KeyType> &head_list,
                     typename TSP::KeyType key) {
  return std::upper_bound(head_list.begin(), head_list.end(), key) - 1;
}

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
  typedef tapas::Region<TSP> Region;
  typedef hot::Cell<TSP> Cell;
  typedef tapas::BodyIterator<Cell> BodyIterator;
  typedef tapas::sfc::Morton<DIM> SFC;
  
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
  typedef tapas::Region<TSP> Region;
  typedef hot::Cell<TSP> Cell;
  typedef tapas::BodyIterator<Cell> BodyIterator;

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

#endif // TAPAS_HOT_
