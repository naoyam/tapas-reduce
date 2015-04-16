/**
 * @file morton_hot.h
 * @brief Implements MPI-based, Morton-order HOT (Hashed Octree) implementation of Tapas's tree
 */
#ifndef TAPAS_MORTON_HOT_
#define TAPAS_MORTON_HOT_

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
#include "tapas/morton_common.h"

#define DEBUG_SENDRECV

namespace tapas {

/**
 * @brief Provides MPI-based distributed Morton-order octree partitioning
 */
namespace morton_hot {

using namespace morton_common;

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
 * @brief Find a (one-to-one or one-to-many) mapping F :: S -> R and returns F(x).
 *
 */
template <class T>
std::vector<T> SendRecvMapping(const std::vector<T> &S, // senders
                               const std::vector<T> &R, // receives
                               const T& x) {
  if (S.size() == 0 || R.size() == 0) {
    return std::vector<T>();
  }

  int s = S.size();
  int r = R.size();
  int n = (r-1) / s + 1; // receivers per senders
  auto pos = std::find(std::begin(S), std::end(S), x);

  if (pos != std::end(S)) {
    // x is a sender
    int i = pos - std::begin(S);
    if (s >= r) {
      if (i < r) {
        return std::vector<T>(1, R[i]);
      } else {
        return std::vector<T>();
      }
    } else {
      if (n*i >= r) {
        return std::vector<T>();
      } else {
        int beg = n * i;
        int end = std::min(n * (i+1), r);
        return std::vector<T>(&R[beg], &R[end]);
      }
    }
  } else {
    // x is a receiver
    int i = std::find(std::begin(R), std::end(R), x) - std::begin(R);
    if (s >= r) {
      return std::vector<int>(1, S[i]);
    } else {
      return std::vector<int>(1, S[i/n]);
    }
  }
}

template <int DIM>
struct HelperNode {
  KeyType key;          //!< Morton key
  Vec<DIM, int> anchor; //!< Morton-key like vector without depth information
  index_t p_index;      //!< Index of the corresponding body
  index_t np;           //!< Number of particles in a node
};

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
  for (size_t i = 0; i < bodies.size(); i++) {
    auto &b = bodies[i];
    strm << std::scientific << std::showpos << b.X[0] << " "
         << std::scientific << std::showpos << b.X[1] << " "
         << std::scientific << std::showpos << b.X[2] << " "
         << std::fixed << std::setw(10) << keys[i]
         << std::endl;
  }
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
std::vector<HelperNode<TSP::Dim>>
CreateInitialNodes(const typename TSP::BT::type *p, index_t np,
                   const Region<TSP> &r);

template <int DIM, class T>
void AppendChildren(KeyType k, T &s);

template <int DIM, class BT>
void SortBodies(const typename BT::type *b, typename BT::type *sorted,
                const HelperNode<DIM> *nodes,
                tapas::index_t nb);

template <int DIM, class BodyType>
void SortBodies2(std::vector<BodyType> &bodies, const std::vector<HelperNode<DIM>> &hn);

template <int DIM>
void CompleteRegion(KeyType x, KeyType y, KeyVector &s);

template <int DIM>
index_t GetBodyNumber(const KeyType k, const HelperNode<DIM> *hn,
                      index_t offset, index_t len);


template <class TSP>
class Partitioner;

template <class TSP> // TapasStaticParams
class Cell: public tapas::BasicCell<TSP> {
  friend class Partitioner<TSP>;
  friend class BodyIterator<Cell>;
  
 public:
  typedef unordered_map<KeyType, Cell*> CellHashTable;
  typedef typename TSP::ATTR attr_type;
  typedef typename TSP::BT::type BodyType;
  typedef typename TSP::BT_ATTR BodyAttrType;
  typedef BodyIterator<Cell> BodyIter;

 public:
  
  template<class T>
  using VecPtr = std::shared_ptr<std::vector<T>>;

  /**
   * @brief Constructor of Cell class
   * @param key Key of the cell
   * @param is_local Whether the cell is a local cell
   * @param is_leaf  Whether the cell is a leaf cell
   * @param body_beg If is_leaf is true, starting index of the range of 
   *                 local_bodies and local_body_keys that the cell owns. 
   *                 Otherwise, 0
   * @param body_num If is_leaf is true, the number of bodies the cell owns.
   * @param ht Hashtable.
   * @param region A Region object.
   * @param leaf_keys array of space filling keys of leaf cells
   * @param leaf_nb array of body numbers of leaf cells
   * @param leaf_owners Process ID (i.e. rank if in MPI) that cells belong to
   * @param local_bodies Array of bodies that the local process owns.
   * @param local_body_keys Keys of local bodies
   * @param local_body_attrs Array of BodyAttrType corresponding to local_body_keys
   */
  Cell(KeyType key,
       bool is_local,
       index_t body_beg, index_t body_num,
       std::shared_ptr<CellHashTable> ht,
       std::shared_ptr<std::mutex> ht_mtx,
       const Region<TSP> &region,
       VecPtr<KeyType>  leaf_keys,       // Keys of all (local and remote) leaf cells
       VecPtr<index_t>  leaf_nb,
       VecPtr<int>      leaf_owners,     // Process IDs that own i-th leaf cell.
       VecPtr<BodyType> local_bodies,    // Bodies which this process owns
       VecPtr<KeyType>  local_body_keys, // Keys of local_bodies
       VecPtr<BodyAttrType> local_body_attrs
       ) :
      tapas::BasicCell<TSP>(CalcRegion(key, region), body_beg, body_num),
      key_(key), is_local_(is_local), is_dummy_(false), region_global_(region),
      ht_(ht), ht_mtx_(ht_mtx),
      leaf_keys_(leaf_keys),
      leaf_nb_(leaf_nb),
      leaf_owners_(leaf_owners),
      local_bodies_(local_bodies),
      local_body_keys_(local_body_keys),
      local_body_attrs_(local_body_attrs),
      owners_(),
      mpi_tag_(GetMpiTag(key))
  {
    CalcOwnerProcesses();

    // Check if I'm a leaf
    is_leaf_ = find(leaf_keys_->begin(), leaf_keys_->end(), key) != leaf_keys_->end();

    // Count number of bodies of this cell (including cells that are indirectly owned bodies)
    if (body_num == 0) {
      index_t beg, end;
      GetDescendantRange<TSP::Dim>(key, leaf_keys_->begin(), leaf_keys->end(), beg, end);
      int nb = 0;
      for (auto i=beg; i < end; i++) {
        nb += (*leaf_nb_)[i];
      }
      this->nb_ = nb;
    }
  }

  // Create a dummy root node
  Cell(std::shared_ptr<CellHashTable> ht, std::shared_ptr<std::mutex> ht_mtx, const Region<TSP> &r) :
      tapas::BasicCell<TSP>(r, 0, 0),
      key_(0), is_local_(false), is_leaf_(false), is_dummy_(true), ht_(ht), ht_mtx_(ht_mtx),
      mpi_tag_(0)
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
  Cell &subcell(int idx) const;

  /**
   * @brief Returns the parent cell if it's local.
   *
   * Returns a reference to the parent cell object of this cell.
   * In this HOT implementation, parent cell of a local cell is
   * always a local cell.
   */
  Cell &parent() const;

  const std::vector<int> &owners() const { return owners_; }

  int depth() const {
    return MortonKeyGetDepth(key_);
  }

#ifdef DEPRECATED
    typename TSP::BT::type &particle(index_t idx) const {
        return body(idx);
    }
#endif
    typename TSP::BT::type &body(index_t idx) const;
    BodyIterator<Cell> bodies() const;
#ifdef DEPRECATED
    typename TSP::BT_ATTR *particle_attrs() const {
        return body_attrs();
    }
#endif
    typename TSP::BT_ATTR *body_attrs() const;
    SubCellIterator<Cell> subcells() const;

 protected:
  // member variables
  KeyType key_; //!< Key of the cell
  bool is_local_;
  bool is_leaf_;
  bool is_dummy_; //!< A dummy cell is returned from Partition if no leaf cell is assigned to the process.
  Region<TSP> region_global_;
  
  std::shared_ptr<CellHashTable> ht_; //!< Hash table of KeyType -> Cell*
  std::shared_ptr<std::mutex>    ht_mtx_; //!< mutex to manipulate ht_

  VecPtr<KeyType>  leaf_keys_;
  VecPtr<index_t>  leaf_nb_;
  VecPtr<int>      leaf_owners_;
  VecPtr<BodyType> local_bodies_;
  VecPtr<KeyType>  local_body_keys_;
  VecPtr<BodyAttrType> local_body_attrs_;
  std::vector<int> owners_; //<! Processes that have this cell locally.
  const int mpi_tag_;
  
  // utility/accessor functions
  Cell *Lookup(KeyType k) const;
  CellHashTable *ht() { return ht_; }
  typename TSP::BT_ATTR &body_attr(index_t idx) const;
  virtual void make_pure_virtual() const {}
  void RegisterCell(Cell<TSP> *c);

  static std::vector<int> CalcOwnerProcsOfCell(KeyType key,
                                               const std::vector<KeyType> &leaf_keys,
                                               const std::vector<int> &leaf_owners);
  void CalcOwnerProcesses();
  static Region<TSP> CalcRegion(KeyType, const Region<TSP>& r);
  
  // Map-related stuff
  void RecvCell(int pid);
  void SendCell(const std::vector<int> pids);
  void ExchangeCell(Cell<TSP> &remote_cell);

  /**
   *
   */
  static int GetMpiTag(KeyType key) {
    // int max_tag = std::numeric_limits<int>::max();
    void *v;
    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &v, &flag);
    assert(flag);

    int max_tag = *(int*)v;
    int val = 1;
    for (size_t i = 0; i < sizeof(KeyType); i++) {
      val = (val + (key & 0xFF)) * 37 % max_tag;
      key >>= 8;
    }

    return val;
  }
  
  /**
   *
   */
  static int GetMpiTag(KeyType k1, KeyType k2) {
    // int max_tag = std::numeric_limits<int>::max();
    void *v;
    int flag;
    MPI_Comm_get_attr(MPI_COMM_WORLD, MPI_TAG_UB, &v, &flag);
    assert(flag);

    if (k1 > k2) std::swap(k1, k2);

    int max_tag = *(int*)v;
    int val = 1;
    for (int i = 0; i < sizeof(KeyType); i++) {
      val = (val + (k1 & 0xFF)) * 37 % max_tag;
      val = (val + (k2 & 0xFF)) * 37 % max_tag;
      k1 >>= 8;
      k2 >>= 8;
    }

    return val;
  }
}; // class Cell




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
std::vector<HelperNode<TSP::Dim>> CreateInitialNodes(const typename TSP::BT::type *bodies,
                                                     index_t nb,
                                                     const Region<TSP> &r) {
    const int Dim = TSP::Dim;
    typedef typename TSP::FP FP;
    typedef typename TSP::BT BT;

    std::vector<HelperNode<Dim>> nodes(nb);
    FP num_cell = 1 << MAX_DEPTH; // maximum number of cells in one dimension
    Vec<Dim, FP> pitch;           // possible minimum cell width
    for (int d = 0; d < Dim; ++d) {
      pitch[d] = (r.max()[d] - r.min()[d]) / num_cell;
    }

#if 0
    // debug print (to be deleted)
    BarrierExec([&](int rank, int size) {
        std::cerr << "CreateInitialNodes: rank " << rank << " pitch = " << pitch << std::endl;
        std::cerr << "CreateInitialNodes: rank " << rank << " r.max = " << r.max() << std::endl;
        std::cerr << "CreateInitialNodes: rank " << rank << " r.min = " << r.min() << std::endl;
      });
#endif

    for (index_t i = 0; i < nb; ++i) {
        // First, create 1 helper cell per particle
        HelperNode<Dim> &node = nodes[i];
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
            if (node.anchor[d] == (1 << MAX_DEPTH)) {
                TAPAS_LOG_DEBUG() << "Particle located at max boundary." << std::endl;
                node.anchor[d]--;
            }
        }
#ifdef TAPAS_DEBUG
        assert(node.anchor >= 0);
# if 1
        if (!(node.anchor < (1 << MAX_DEPTH))) {
            TAPAS_LOG_ERROR() << "Anchor, " << node.anchor
                              << ", exceeds the maximum depth." << std::endl
                              << "Particle at "
                              << ParticlePosOffset<Dim, FP, BT::pos_offset>::vec((const void*)&(bodies[i]))
                              << std::endl;
            TAPAS_DIE();
        }
# else
        assert(node.anchor < (1 << MAX_DEPTH));
# endif
#endif // TAPAS_DEBUG

        node.key = CalcFinestMortonKey<Dim>(node.anchor);
    }

    return nodes;
}

template <int DIM, class BT>
void SortBodies(const typename BT::type *b, typename BT::type *sorted,
                const HelperNode<DIM> *sorted_nodes,
                tapas::index_t nb) {
    for (index_t i = 0; i < nb; ++i) {
        sorted[i] = b[sorted_nodes[i].p_index];
    }
}

/**
 * \brief Sort bodies according to already-sorted HelperNodes.
 * 
 * let p = hn[i].p_index. p indicates that p-th body in bodies should be at i-th after sorting.
 * hn is already sorted according to their space filling keys.
 */
template <int DIM, class BodyType>
void SortBodies2(std::vector<BodyType> &bodies, const std::vector<HelperNode<DIM>> &hn) {
  assert(bodies.size() == hn.size());
  index_t nb = bodies.size();
  for (index_t i = 0; i < nb; i++) {
    index_t bi = hn[i].p_index;
    if (bi > i) {
      std::swap(bodies[i], bodies[bi]);
    }
  }
}



template <int DIM, class T>
void AppendChildren(KeyType x, T &s) {
    int x_depth = MortonKeyGetDepth(x);
    int c_depth = x_depth + 1;
    if (c_depth > MAX_DEPTH) return;
    x = MortonKeyIncrementDepth(x, 1);
    for (int i = 0; i < (1 << DIM); ++i) {
      KeyType child_key = ((KeyType)i << ((MAX_DEPTH - c_depth) * DIM + DEPTH_BIT_WIDTH));
      s.push_back(x | child_key);
      TAPAS_LOG_DEBUG() << "Adding child " << (x | child_key) << std::endl;
    }
}

template <int DIM>
void CompleteRegion(KeyType x, KeyType y,
                    KeyVector &s) {
    KeyType fa = FindFinestAncestor<DIM>(x, y);
    KeyList w;
    AppendChildren<DIM>(fa, w);
    tapas::PrintKeys(w, std::cout);
    while (w.size() > 0) {
        KeyType k = w.front();
        w.pop_front();
        TAPAS_LOG_DEBUG() << "visiting " << k << std::endl;
        if ((k > x && k < y) && !MortonKeyIsDescendant<DIM>(k, y)) {
            s.push_back(k);
            TAPAS_LOG_DEBUG() << "Adding " << k << " to output set" << std::endl;
        } else if (MortonKeyIsDescendant<DIM>(k, x) ||
                   MortonKeyIsDescendant<DIM>(k, y)) {
            TAPAS_LOG_DEBUG() << "Adding children of " << k << " to work set" << std::endl;
            AppendChildren<DIM>(k, w);

        }
    }
    std::sort(std::begin(s), std::end(s));
}

/**
 * @brief 1-parameter Map function for 
 */
template <class TSP>
void Cell<TSP>::Map(Cell<TSP> &cell, std::function<void(Cell<TSP>&)> f) {
  if (cell.IsDummy()) {
    return;
  }

  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (SimplifyKey(cell.key()) == "0461...7906") {
    Stderr e("debug");
    e.out() << cell.key() << " Map() is called." << std::endl;
    e.out() << cell.key() << " time= " << (long)(MPI_Wtime()*1000)%100000/1000.0 << std::endl;
    e.out() << cell.key() << " object address = " << &cell << std::endl;
    e.out() << cell.key() << " :           M = " << cell.attr().M << std::endl;
  }
  
  if (cell.is_local_) {
    f(cell);

    // Processes that needs the cell information can be identified by calculating the difference of sets
    //   {parent's owners} - {it's owners}
    
    if (cell.key() != 0) { // if not root (the root cell is never remote)
      // Owner processes of the remote cell can be obtained by
      // by {parent's owners} - {the cell's owners}
      std::vector<int> senders = cell.owners(); // Owner of the cells
      std::vector<int> recvers = SetDiff(cell.parent().owners(), senders); // Requesters
      std::vector<int> peers = SendRecvMapping(senders, recvers, rank);
      
      cell.SendCell(peers);
    }
  } else {
    // root cell is never 'remote' because it is shared by all processes
    assert(cell.key() != 0);
    
    // The cells is remote, which means the process must obtain cell information
    // from a process(sender) that owns the cell.
    
    // Senders = Owner processes
    std::vector<int> senders = cell.owners(); // Owner of the cells

    // Receivers = {pcell's owners} - {the cell's owners}
    // This process is a receiver of the current cell
    std::vector<int> recvers = SetDiff(cell.parent().owners(), senders);

    auto peers = SendRecvMapping(senders, recvers, rank);
    assert(peers.size() == 1);
    
    cell.RecvCell(peers[0]); // the sender finishes applying f first, then send the cell info to this process.
    usleep(100000);
    if (SimplifyKey(cell.key()) == "0461...7906") {
      Stderr e("debug");
      e.out() << cell.key() << " RecvCell finished." << std::endl;
      e.out() << cell.key() << " time= " << (long)(MPI_Wtime()*1000)%100000/1000.0 << std::endl;
      e.out() << cell.key() << " object address = " << &cell << std::endl;
      e.out() << cell.key() << " :           M = " << cell.attr().M << std::endl;
    }
  }
}

// A special-purpose struct to exchange cell information
template <class TSP>
struct CellInfoBinder {
  KeyType key;
  typename TSP::ATTR cell_attr;
  index_t num_bodies;
  bool is_leaf;
  int check;
};

/**
 * @brief Exchange cell information between the local process and a remote process
 * 
 * Exchange *this cell, which is local, and the remote cell.
 */
template<class TSP>
void Cell<TSP>::ExchangeCell(Cell<TSP> &remote) {
  assert(this->IsLocal() && !remote.IsLocal());
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  Stderr e("exchangecell");

  std::vector<int> local_owners = this->owners();
  std::vector<int> remote_owners = remote.owners();
  
  // this process sends *this cell to some remote processes
  std::vector<int> send_to = SendRecvMapping(local_owners, remote_owners, rank);
  std::vector<int> recv_from = SendRecvMapping(remote_owners, local_owners, rank);

  if (SimplifyKey(this->key()) == "0345...0929") {
    e.out() << "*** I have 0345...0929" << std::endl;
  }
  
  int mpi_tag = Cell::GetMpiTag(this->key(), remote.key());

  e.out() << "Exchanging cells " << SimplifyKey(this->key()) << " && " << SimplifyKey(remote.key()) << std::endl;
  e.out() << SimplifyKey(this->key()) << "'s owners = ";
  for (auto &&o : local_owners) e.out() << o << " ";
  e.out() << " (" << this->owners().size() << " elements)";
  e.out() << std::endl;
  
  e.out() << SimplifyKey(remote.key()) << "'s owners = ";
  for (auto &&o : remote_owners) e.out() << o << " ";
  e.out() << std::endl;
  
  e.out() << "tag = " << mpi_tag << std::endl;
  e.out() << "send " << SimplifyKey(this->key()) << " to = ";
  for (auto && i: send_to) e.out() << i << " ";
  e.out() << std::endl;
  
  e.out() << "recv " << SimplifyKey(remote.key()) << " from = ";
  for (auto && i: recv_from) e.out() << i << " ";
  e.out() << std::endl;
    
  CellInfoBinder<TSP> send_data = {
    key_,
    this->attr(),
    this->nb(),
    this->IsLeaf(),
    rand() // check
  };
  
  std::vector<MPI_Status>  stats(send_to.size());
  std::vector<MPI_Request> reqs(send_to.size());

  for (int i = 0; i < send_to.size(); i++) {
    // Send the local cell to the remote node(s) asynchornously
    Stderr stderr("send");
    stderr.out() << "SendCell: cell=" << SimplifyKey(key_) << " "
                 << "I'm=" << rank << " "
                 << "dst=" << send_to[i] << " "
                 << "IsLeaf=" << send_data.is_leaf << " " 
                 << "check=" << send_data.check << " "
                 << "tag=" << mpi_tag_ << " "
                 << std::endl;
    MPI_Isend(&send_data, sizeof(send_data), MPI_BYTE, send_to[i], mpi_tag, MPI_COMM_WORLD, &reqs[i]);
  }

  e.out() << "Isend done" << std::endl;

  // Receive data from one of the owners fo the remote cell
  CellInfoBinder<TSP> recv_data;
  MPI_Status recv_stat;
  int ret = MPI_Recv(&recv_data, sizeof(recv_data), MPI_BYTE, recv_from[0], mpi_tag, MPI_COMM_WORLD, &recv_stat);
  assert(ret == MPI_SUCCESS);
  if (recv_data.key != remote.key()) {
    std::cerr << "tag = " << mpi_tag << std::endl;
    assert(recv_data.key == remote.key());
  }

  remote.attr_    = recv_data.cell_attr;
  remote.nb_      = recv_data.num_bodies;
  remote.is_leaf_ = recv_data.is_leaf;

  e.out() << "Recv done." << std::endl;

  MPI_Waitall(reqs.size(), reqs.data(), stats.data());
  e.out() << "Waitall done." << std::endl;
  e.out() << std::endl;
}

/**
 * 2-parameter Map function over cells (apply user function to products of cells)
 */
template<class TSP>
void Cell<TSP>::Map(Cell<TSP> &c1, Cell<TSP> &c2,
                    std::function<void(Cell<TSP>&, Cell<TSP>&)> f) {
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  {
    Stderr e("map2");
    e.out() << "Map: "
            << SimplifyKey(c1.key()) << ", "
            << SimplifyKey(c2.key()) << " : "
            << (c1.IsLocal() ? "local" : "remote") << ","
            << (c2.IsLocal() ? "local" : "remote")
            << std::endl;
    ;
  }

  if (!c1.IsLocal() && !c2.IsLocal()) return;

  mk_task_group;

  if (c1.IsLocal()) {
    // Send cell information to necessary processes
    // recvers : processes that has c2 but not c1, and will require c1 information
    // senders : processes that have c1 locally and can send c1 information to other processes.
    // send_to : Assignment of the local process 
    std::vector<int> recvers = SetDiff(c2.owners(), c1.owners());
    std::vector<int> senders = c1.owners();
    std::vector<int> send_to = SendRecvMapping(senders, recvers, rank);
    {
      Stderr e("map2");
      e.out() << "c1 [" << SimplifyKey(c1.key()) << "] is local. (I'm " << rank << ")" << std::endl;
      e.out() << "c1.owners() = " << "[" << join(" ", c1.owners()) << "]" << std::endl;
      e.out() << "c2.owners() = " << "[" << join(" ", c2.owners()) << "]" << std::endl;
      e.out() << "recvers = " << "[" << join(" ", recvers) << "]" << std::endl;
      e.out() << "senders = " << "[" << join(" ", senders) << "]" << std::endl;
      e.out() << "send_to = " << "[" << join(" ", send_to) << "]" << std::endl;
      e.out() << std::endl;
    }
    auto task = [&c1, send_to]() { c1.SendCell(send_to); };
    create_taskc(task);
  }
  
  if (c2.IsLocal()) {
    std::vector<int> recvers = SetDiff(c1.owners(), c2.owners());
    std::vector<int> senders = c2.owners();
    std::vector<int> send_to = SendRecvMapping(senders, recvers, rank);
    {
      Stderr e("map2");
      e.out() << "c2 [" << SimplifyKey(c2.key()) << "] is local. (I'm " << rank << ")" << std::endl;
      e.out() << "c2.owners() = " << "[" << join(" ", c2.owners()) << "]" << std::endl;
      e.out() << "c1.owners() = " << "[" << join(" ", c1.owners()) << "]" << std::endl;
      e.out() << "recvers = "     << "[" << join(" ", recvers) << "]" << std::endl;
      e.out() << "senders = "     << "[" << join(" ", senders) << "]" << std::endl;
      e.out() << "send_to = "     << "[" << join(" ", send_to) << "]" << std::endl;
      e.out() << std::endl;
    }
    auto task = [&c2, send_to]() { c2.SendCell(send_to); };
    create_taskc(task);
  }
  
  if (!c1.IsLocal()) {
    std::vector<int> recvers = SetDiff(c2.owners(), c1.owners());
    std::vector<int> senders = c1.owners();
    std::vector<int> recv_from = SendRecvMapping(senders, recvers, rank);
    Stderr e("map2");
    e.out() << "Requesting " << SimplifyKey(c1.key()) << " to " << recv_from[0] << std::endl;
      
    c1.RecvCell(recv_from[0]);
  }
  
  if (!c2.IsLocal()) {
    std::vector<int> recvers = SetDiff(c1.owners(), c2.owners());
    std::vector<int> senders = c2.owners();
    std::vector<int> recv_from = SendRecvMapping(senders, recvers, rank);
    
    Stderr e("map2");
    e.out() << "Requesting " << SimplifyKey(c2.key()) << " to " << recv_from[0] << std::endl;
    
    c2.RecvCell(recv_from[0]);
  }

  wait_tasks;

  {
    Stderr e("map2");
    e.out() << "Map: "
            << SimplifyKey(c1.key()) << ", "
            << SimplifyKey(c2.key()) << " : "
            << "done." << std::endl;
  }
  
  f(c1, c2);
}

template <class TSP>
void Cell<TSP>::RecvCell(int pid) {
  // Receive Cell/BodyAttr info from the process(pid).
  //   Cell attributes
  //   Bodies (if the cell is a leaf)
  //   Body Attributes (if the cell is a leaf)
  
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Stderr stderr("recv");
    stderr.out() << "RecvCell: cell=" << SimplifyKey(key_) << " "
                 << "me=" << rank << " "
                 << "src=" << pid << " "
                 << "tag=" << std::setw(10) << mpi_tag_
                 << std::endl;
  }
  
  // First, request CellAttr data. On sender side (in SendCell)
  // note that this MPI_recv() call is handled by MPI_Isend in SendCell() in proc pid.

  // First, call MPI_Probe to get the message size.
  MPI_Status stat;
  int ret = MPI_Probe(pid, mpi_tag_, MPI_COMM_WORLD, &stat);
  assert(ret == MPI_SUCCESS);

  int bytes; // received size in bytes
  MPI_Get_count(&stat, MPI_BYTE, &bytes);

  assert(bytes % sizeof(CellInfoBinder<TSP>) == 0);

  CellInfoBinder<TSP> *data = new CellInfoBinder<TSP>[bytes/sizeof(CellInfoBinder<TSP>)];
  
  ret = MPI_Recv(data, bytes, MPI_BYTE, pid, mpi_tag_, MPI_COMM_WORLD, &stat);
  assert(ret == MPI_SUCCESS);

  assert(data[0].key == key_);
  
  //assert(stat.MPI_ERROR == MPI_SUCCESS);
  this->attr()   = data[0].cell_attr;
  this->nb_      = data[0].num_bodies;
  this->is_leaf_ = data[0].is_leaf;

  // set the bodies
  BodyType *bodies = reinterpret_cast<BodyType*>(&data[1]);
  this->bid_ = 0;
  this->local_bodies_ = VecPtr<BodyType>(new std::vector<BodyType>(bodies, bodies + this->nb_));

  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    Stderr stderr("recv");
    stderr.out() << "RecvCell: cell=" << SimplifyKey(key_) << " "
                 << "me="     << rank << " "
                 << "src="    << pid << " "
                 << "tag="    << std::setw(10) << mpi_tag_ << " "
                 << "IsLeaf=" << data[0].is_leaf << " "
                 << "check="  << data[0].check << " "
                 << "R="      << this->attr().R << " "
                 << "done "
                 << std::endl;
  }
}

template <class TSP>
void Cell<TSP>::SendCell(std::vector<int> remote_pids) {
  typedef typename TSP::BT::type BodyType;
  
  // This function does not take a reference because may be called in an independent thread (task)
  // and the original vector might be destroyed.
  if (remote_pids.size() == 0) {
    return;
  }

  // Send cell attributes and bodies (if the cell is leaf) to the remote pids.
  // To send both of cell attrs and bodies, array of binder is used.
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // note that this->nb() is the total number of bodies in the cell, even if the cell is not leaf
  index_t num_bodies = this->nb();
  index_t size_bodies = sizeof(BodyType) * num_bodies;

  // length of total data to be sent.
  // the data is allocated as an array of CellInfoBinder<TSP>.
  // len is the minimal length that can accomodate cell attributes and bodies.
  index_t len = (size_bodies - 1) / sizeof(CellInfoBinder<TSP>) + 2;
  CellInfoBinder<TSP> *data = new CellInfoBinder<TSP>[len];
  
  data[0].key = key_;
  data[0].cell_attr = this->attr();
  data[0].num_bodies = num_bodies;
  data[0].is_leaf = this->IsLeaf(); 
  data[0].check = rand(); // for debugging. to be deleted.

  BodyType *bodies = reinterpret_cast<BodyType*>(&data[1]);
  index_t bid = this->bid();

  for (int bi = 0; bi < num_bodies; bi++) {
    bodies[bi] = local_bodies_->at(bi + bid);
  }

  MPI_Status *stats = new MPI_Status[remote_pids.size()];
  MPI_Request *reqs = new MPI_Request[remote_pids.size()];
  
  for (size_t i = 0; i < remote_pids.size(); i++) {
    {
      Stderr stderr("send");
      stderr.out() << "SendCell: cell=" << SimplifyKey(key_) << " "
                   << "I'm="    << rank << " "
                   << "dst="    << remote_pids[i] << " "
                   << "nb="     << num_bodies << " "
                   << "sizeof(CellInfoBinder)=" << sizeof(CellInfoBinder<TSP>) << " "
                   << "sizeof(BodyType)=" << sizeof(BodyType) << " "
                   << "sizeof(BodyType)*" << num_bodies << "=" << sizeof(BodyType) * num_bodies << " "
                   << "len="  << len << " "
                   << "IsLeaf=" << data[0].is_leaf << " "
                   << "check="  << data[0].check << " "
                   << "tag="    << mpi_tag_ << " "
                   << std::endl;
    }
    MPI_Isend(data, sizeof(data[0]) * len, MPI_BYTE, remote_pids[i], mpi_tag_, MPI_COMM_WORLD, &reqs[i]);
  }

  MPI_Waitall(remote_pids.size(), reqs, stats);
  delete[] stats;
  delete[] reqs;
  
  for (size_t i = 0; i < remote_pids.size(); i++) {
    Stderr stderr("send");
    stderr.out() << "SendCell: cell=" << SimplifyKey(key_) << " "
                 << "I'm="    << rank << " "
                 << "dst="    << remote_pids[i] << " "
                 << "IsLeaf=" << data[0].is_leaf << " " 
                 << "check="  << data[0].check << " "
                 << "tag="    << mpi_tag_ << " "
                 << "done"    << std::endl;
  }
  delete[] data;
}

template <class TSP>
bool Cell<TSP>::operator==(const Cell &c) const {
    return key_ == c.key_;
}

template <class TSP>
bool Cell<TSP>::IsRoot() const {
    return MortonKeyGetDepth(key_) == 0;
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
Cell<TSP> &Cell<TSP>::subcell(int idx) const {
  if (IsLeaf()) {
    TAPAS_LOG_ERROR() << "Trying to access children of a leaf cell." << std::endl;
    TAPAS_DIE();
  }


  KeyType child_key = MortonKeyChild<TSP::Dim>(key_, idx);
  Cell *c = Lookup(child_key);
  
  if (c == nullptr) {
    std::lock_guard<std::mutex> lock(*ht_mtx_);
    c = Lookup(child_key);
    if (c == nullptr) {
      // leaf if if key is contained in leaf_keys_

      c = new Cell(child_key, false, 0, 0, ht_, ht_mtx_, region_global_,
                   leaf_keys_, leaf_nb_, leaf_owners_,
                   local_bodies_, local_body_keys_, local_body_attrs_);
      (*ht_)[child_key] = c;
    }
  }
  
  return *c;
}

template <class TSP>
Cell<TSP> *Cell<TSP>::Lookup(KeyType k) const {
    auto i = ht_->find(k);
    if (i != ht_->end()) {
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
    KeyType parent_key = MortonKeyParent<TSP::Dim>(key_);
    auto *c = Lookup(parent_key);
    if (c == nullptr) {
      TAPAS_LOG_ERROR() << "Parent (" << parent_key << ") of "
                        << "cell (" << key_ << ") not found.\n"
                        << "Parent key = " << MortonKeyDecode<TSP::Dim>(parent_key) << "\n"
                        << "Child key =  " << MortonKeyDecode<TSP::Dim>(key_)
                        << std::endl;
      TAPAS_DIE();
    }
    return *c;
}

template <class TSP>
typename TSP::BT::type &Cell<TSP>::body(index_t idx) const {
  assert(idx < this->nb());
  return local_bodies_->at(this->bid() + idx);
}

template <class TSP>
typename TSP::BT_ATTR *Cell<TSP>::body_attrs() const {
  return local_body_attrs_->data();
}

template <class TSP>
typename TSP::BT_ATTR &Cell<TSP>::body_attr(index_t idx) const {
  assert(idx < this->nb());
  return local_body_attrs_->at(this->bid() + idx);
}

template <class TSP>
SubCellIterator<Cell<TSP>> Cell<TSP>::subcells() const {
    return SubCellIterator<Cell>(*this);
}

template <class TSP>
BodyIterator<Cell<TSP>> Cell<TSP>::bodies() const {
    return BodyIterator<Cell<TSP> >(*this);
}

template <class TSP> // Tapas static params
class Partitioner {
  private:
    const int max_nb_;

  public:
    Partitioner(unsigned max_nb): max_nb_(max_nb) {}

    Cell<TSP> *Partition(typename TSP::BT::type *b, index_t nb,
                         const Region<TSP> &r);
    Cell<TSP> *Partition(std::vector<typename TSP::BT::type> &b,
                         const Region<TSP> &r);
  private:
    void Refine(Cell<TSP> *c,
                const std::vector<HelperNode<TSP::Dim>> &hn,
                const typename TSP::BT::type *b,
                int cur_depth,
                KeyType cur_key) const;
}; // class Partitioner

/**
 * @brief Overloaded version of Partitioner::Partition
 */
template <class TSP>
Cell<TSP>*
Partitioner<TSP>::Partition(std::vector<typename TSP::BT::type> &b, const Region<TSP> &r) {
    return Partitioner<TSP>::Partition(b.data(), b.size(), r);
}

/**
 * @brief Returns a std::vector of parent's children
 */
template<int DIM>
std::vector<KeyType> GetChildren(KeyType parent) {
  std::vector<KeyType> ret;

  KeyType child_key = MortonKeyFirstChild<DIM>(parent);
  for (int child_idx = 0; child_idx < (1<<DIM); child_idx++) {
    ret.push_back(child_key);
    child_key = CalcMortonKeyNext<DIM>(child_key);
  }

  return ret;
}

/**
 * @brief Split cells that have more than nb_max bodies (not recursive)
 *
 * @param cell_keys Array of morton keys of cells
 * @param nb Array of current numbers of bodies of the cells
 * @param max_nb Criteria to split a cell
 */
template<int DIM>
std::vector<KeyType> SplitLargeCellsOnce(const std::vector<KeyType> &cell_keys,
                                         const std::vector<index_t> &nb,
                                         int max_nb) {
  std::vector<KeyType> ret; // new

  for (size_t i = 0; i < cell_keys.size(); i++) {
    if (nb[i] <= max_nb) {
      // This cell does not need to be split.
      ret.push_back(cell_keys[i]);
    } else {
      // Create 2^DIM children (8 in 3-dim)
      auto children = GetChildren<DIM>(cell_keys[i]);
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


/**
 * @brief Calculate which processes own the cell.
 */
template <class TSP>
void Cell<TSP>::CalcOwnerProcesses() {
  this->owners_ = CalcOwnerProcsOfCell(key_, *leaf_keys_, *leaf_owners_);
}

template<class TSP>
void Cell<TSP>::RegisterCell(Cell<TSP> *c) {
  std::lock_guard<std::mutex> lock(*ht_mtx_);
  (*ht_)[c->key()] = c;
}

template <class TSP>
Region<TSP> Cell<TSP>::CalcRegion(KeyType key, const Region<TSP> &region) {
  const int kDim = TSP::Dim;
  const int kMask = (1 << TSP::Dim) - 1;
  
  Stderr err("center");

  auto r = region;
  int depth = MortonKeyGetDepth(key);
  KeyType key_body = MortonKeyRemoveDepth(key);
  
  err.out() << SimplifyKey(key) << " "
            << depth << " "
            << r << " --> ";

  for (int d = 0; d < depth; d++) {
    int direction = (key_body >> (kDim * (MAX_DEPTH - d - 1))) & kMask;
    r = r.PartitionBSP(direction);
    key >>= TSP::Dim;
    //err.out() << d << ":" << direction << " " << r << " ";
  }
  
  err.out() << r << std::endl;
  
  return r;
}

template <class TSP>
std::vector<int> Cell<TSP>::CalcOwnerProcsOfCell(KeyType key,
                                                 const std::vector<KeyType> &leaf_keys,
                                                 const std::vector<int> &leaf_owners) {
  // Find the range of leaf cells that are under the cell
  auto beg = std::begin(leaf_keys);
  auto end = std::end(leaf_keys);
  KeyType key_next = CalcMortonKeyNext<TSP::Dim>(key);
  index_t owner_beg = std::lower_bound(beg, end, key) - beg;
  index_t owner_end = std::lower_bound(beg, end, key_next) - beg;

  GetDescendantRange<TSP::Dim>(key, beg, end, owner_beg, owner_end);

  if (SimplifyKey(key) == "0230...3954") {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::stringstream ss;
    ss << rank << " " << SimplifyKey(key) << " "
       << "owner_beg = " << owner_beg << ","
       << "owner_beg = " << owner_end << " ";
    for (auto i=owner_beg; i < owner_end; i++) {
      ss << leaf_owners[i] << " ";
    }
    ss << "  leaf_keys.size() = " << leaf_keys.size();
    std::cerr << ss.str() << std::endl;
  }

  // Since each process builds a local tree in a bottom-up way,
  // if a process P ownes any of leaf cells uncer the cell, P owns the cell.
  return uniq<int>(std::begin(leaf_owners) + owner_beg,
                   std::begin(leaf_owners) + owner_end);
}


/**
 * @brief Partition the simulation space and build Morton-key based octree
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
  const int Dim = TSP::Dim;
  typedef typename TSP::FP FP;
  typedef typename TSP::BT BT;
  typedef typename TSP::BT_ATTR BodyAttrType;
  typedef typename BT::type BodyType;
  typedef Cell<TSP> CellType;
  typedef HelperNode<Dim> HN;
  
  auto r = ExchangeRegion(reg);

  // Sort local bodies using Morton keys
  std::vector<HN> hn = CreateInitialNodes<TSP>(b, num_bodies, r);
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

  std::vector<KeyType> leaf_keys;   // Morton keys of leaf cells (global)
  std::vector<index_t> leaf_nb_local;  // Number of local bodies in leaf cell[i] (all global cells)
  std::vector<index_t> leaf_nb_global; // Number of global bodies in leaf cell[i] (all global cells)

  // MPI rank and size
  // TOOD: to be replaced by an appropriate abstraction of communication component.
  int rank, size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Start from a root cell and refine it recursively until all cells have at most
  leaf_keys.push_back(0);

  // Loop until all leaf cells have at most max_nb_ bodies.
  while(1) {
    leaf_nb_local.clear();
    leaf_nb_global.clear();
    leaf_nb_local.resize(leaf_keys.size(), 0);
    leaf_nb_global.resize(leaf_keys.size(), 0);

    for (size_t i = 0; i < leaf_keys.size(); i++) {
      // Count process-local bodies belonging to the cell[i].
      leaf_nb_local[i] = GetBodyRange<Dim>(leaf_keys[i], hn, [](const HN &hn) { return hn.key; }).second;
    }

#if 0
    BarrierExec([&] (int rank, int size) {
        std::cerr << "rank " << std::fixed << std::setw(3) << std::left << rank << "  ";
        std::cerr << "hn.size() = " << hn.size() << ", ";
        for (auto n : hn) {
          std::cerr << std::fixed << std::setw(6) << MortonKeyRemoveDepth(n.key) << " ";
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
      leaf_keys = SplitLargeCellsOnce<Dim>(leaf_keys, leaf_nb_global, max_nb_);
    }
  } // end of while(1) loop

  // distribute the morton-ordered leaf cells over processes
  // so that each process has roughly equal number of bodies.
  // Split the morton-ordred curve and assign cells to processes
  std::vector<int> leaf_owners = SplitKeysSimple(leaf_nb_global, size);

  // Exchange bodies using MPI_Alltoallv
  std::vector<int> send_counts(size, 0); // number of bodies that this process sends to others
  std::vector<int> recv_counts(size, 0); // number of bodies that this process receives from others
  
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
  // hn (array of HelperNode) is already sorted by their Morton keys,
  // which also means they are sorted by their parent process.
  std::vector<BodyType> send_bodies(num_bodies);
  std::vector<KeyType>  send_keys(num_bodies);
  for (size_t hi = 0; hi < hn.size(); hi++) {
    int bi = hn[hi].p_index;
    send_bodies[hi] = b[bi];
    send_keys[hi] = hn[hi].key;
  }

  // Calculate send displacements, which is prefix sum of send_bytes_bodies
  std::vector<int> send_disp_bodies(size, 0);
  std::vector<int> send_disp_keys(size, 0);

  for (int p = 0; p < size; p++) {
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

  // Prepare recv_bodies (array of received body, which will be used in actual computation)
  // and recv_keys (array of keys corresponding to recv_bodies)
  // recv_bodies will be held by cells and continuously used after this function.
  int num_bodies_recv = sum(recv_counts); // note: recv_count is in bytes.
  auto recv_bodies = std::make_shared<std::vector<BodyType>>(num_bodies_recv);
  std::vector<KeyType> recv_keys(num_bodies_recv);

  // Calculate recv displacement, which is prefix sum of recv_bytes_bodies and recv_bytes_keys
  std::vector<int> recv_disp_bodies(size, 0);
  std::vector<int> recv_disp_keys(size, 0);
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
                recv_bodies->data(), recv_bytes_bodies.data(), recv_disp_bodies.data(), MPI_BYTE,
                MPI_COMM_WORLD);

  // Call MPI_Alltoallv() for body keys
  MPI_Alltoallv(send_keys.data(), send_bytes_keys.data(), send_disp_keys.data(), MPI_BYTE,
                recv_keys.data(), recv_bytes_keys.data(), recv_disp_keys.data(), MPI_BYTE,
                MPI_COMM_WORLD);

  // Now we have all bodies & keys transferred to their owner processes.
  // Sort the bodies locally using their keys.
  SortByPermutations(recv_keys, *recv_bodies);

  // Dump local bodies into a file named exch_bodies.dat
  // All processes dump bodies in the file in a coordinated way. init_bodies.dat and
  // exch_bodies.dat must match (if sorted).
  BarrierExec([&](int rank, int size) {
      std::stringstream ss;
      ss << "exch_bodies." << size << ".dat";
      bool append_mode = (rank > 0);
      DumpToFile(*(recv_bodies.get()), recv_keys, ss.str().c_str(), append_mode);
    });

  // CellHashTable (which is std::unordered_map<KeyType, CellType*>
  auto ht = std::make_shared<typename CellType::CellHashTable>();
  auto ht_mtx = std::make_shared<std::mutex>();

  // construct a local tree from cells which belong to this process.
  auto leaf_beg = std::lower_bound(std::begin(leaf_owners), std::end(leaf_owners), rank) - std::begin(leaf_owners);
  auto leaf_end = std::upper_bound(std::begin(leaf_owners), std::end(leaf_owners), rank) - std::begin(leaf_owners);

  // heap copy of vectors (shared by Cells)
  auto leaf_owners2     = std::make_shared<std::vector<int>>(leaf_owners);
  auto leaf_keys2       = std::make_shared<std::vector<KeyType>>(leaf_keys);
  auto leaf_nb_global2  = std::make_shared<std::vector<index_t>>(leaf_nb_global);
  auto local_body_keys  = std::make_shared<std::vector<KeyType>>(recv_keys);
  auto local_body_attrs = std::make_shared<std::vector<BodyAttrType>>(num_bodies_recv);

  bzero(reinterpret_cast<void*>(&local_body_attrs->at(0)),
        sizeof(BodyAttrType) * local_body_attrs->size());

  std::vector<Cell<TSP>*> interior_cells;
  Stderr e("partition");

  // Build a local tree in a bottom-up manner.
  for (auto i = leaf_beg; i < leaf_end; i++) {
    KeyType k = leaf_keys[i];
    //KeyType kn = CalcMortonKeyNext<Dim>(k);

    // Find bodies owned by the Cell whose key is k.
    index_t bbeg, bend;
    FindRangeByKey<TSP>(recv_keys, k, bbeg, bend);
    
    // Create a leaf cell
    CellType *c = new CellType(k,               // key
                               true,            // is_local
                               bbeg,            // body index
                               bend - bbeg,     // #bodies
                               ht, ht_mtx,      // CellHashTable & its mutex
                               r,               // Region region
                               leaf_keys2,
                               leaf_nb_global2,
                               leaf_owners2,
                               recv_bodies,     // local bodies
                               local_body_keys, // local body keys
                               local_body_attrs);     // body attrs
    (*ht)[k] = c;
    assert(c->IsLocal() && c->IsLeaf());

    Stderr e("check0001");

    // Create anscestors of the cell c (in a recursive upward way)
    while(1) {
      k = MortonKeyParent<Dim>(k);
      int dp = MortonKeyGetDepth(k);

      if (ht->count(k) == 0) {
        index_t bbeg, bend;
        //FindRangeByKey<TSP>(recv_keys, k, bbeg, bend);
        FindRangeByKey<TSP>(leaf_keys, k, bbeg, bend);
        int nb = 0;
        for (auto i = bbeg; i < bend; i++) {
          nb += leaf_nb_global[i];
        }
        
        if (k == 1) {
          e.out() << "key=" << k << std::endl;
          e.out() << "nb = " << nb << std::endl;
        }
        
        // Create interior cellls (anscestors)
        // Note: if a cell is non-leaf, then bbeg (body begin index) is not correct.
        //       This is because bodies are help only by a process that owns the corresponding leaf cells.
        CellType *c = new CellType(k,                 // key
                                   true,              // is_local
                                   0, nb,             // start index of bodies and numBodies. read the note above
                                   ht, ht_mtx,        // CellHashTable
                                   r,                 // Region region
                                   leaf_keys2,
                                   leaf_nb_global2,
                                   leaf_owners2,
                                   recv_bodies,       // local bodies
                                   local_body_keys,   // local body keys
                                   local_body_attrs); // body attrs
        (*ht)[k] = c;
        interior_cells.push_back(c);
        assert(c->IsLocal() && !c->IsLeaf());
      } else {
        break; // if c's parent is found: all of the ancestors have already been created.
      }
      
      if (dp == 0) break; // stop if k is the root cell;
    }
  }
  // we have created all local cells

  // Debug
  // Dump all (local) cells to a file
  {
    Stderr e("cells");
    for (auto&& iter : (*ht)) {
      KeyType k = iter.first;
      Cell<TSP> *c = iter.second;
      if (c->IsLocal() && c->key() != 0) {
        e.out() << SimplifyKey(k) << " "
                << "d=" << MortonKeyGetDepth(k) << " "
                << "leaf=" << c->IsLeaf() << " "
                << "owners=" << std::setw(2) << std::right << join(",", c->owners()) << " "
                << "nb=" << std::setw(3) << c->nb() << " "
                << "center=[" << c->center() << "] "
                << "next_key=" << SimplifyKey(CalcMortonKeyNext<Dim>(k)) << " "
                << "parent=" << SimplifyKey(MortonKeyParent<Dim>(k)) << " "
                << std::endl;
        // Print bodies which belong to Cell c
        if (c->IsLeaf()) {
          index_t body_beg, body_end;
          FindRangeByKey<TSP>(recv_keys, k, body_beg, body_end);
          for (int i = body_beg; i < body_end; i++) {
            e.out() << "\t\t\t| "
                    << SimplifyKey(recv_keys[i]) << ": "
                    << (*recv_bodies)[i].X
                    << std::endl;
          }
        }
      }
    }
  }
  
  if ((*ht)[0] == nullptr) {
    // Create a dummy cell if ht[0] is nullptr, which means the local process
    // was assigned no leaf cell (but need to return a root cell).
    (*ht)[0] = new CellType(ht, ht_mtx, r);
  }
  
  // return the root cell (root key is always 0)
  return (*ht)[0];

  //-------------

#if 0
  BodyType *b_work = new BodyType[num_bodies];

  // Sort particles to the same order of hn
  SortBodies<Dim, BT>(b, b_work, hn.data(), hn.size());

  std::memcpy(b, b_work, sizeof(BodyType) * num_bodies);
  //BodyAttrType *attrs = new BodyAttrType[nb];
  BodyAttrType *attrs = (BodyAttrType*)calloc(num_bodies, sizeof(BodyAttrType));

  KeyType root_key = 0;
  KeyPair kp = GetBodyRange<Dim>(root_key, hn,
                                 [](const HelperNode<Dim> &hn) { return hn.key; });
  assert(kp.first == 0 && kp.second == num_bodies); // it is root cell, which owns all bodies.
  TAPAS_LOG_DEBUG() << "Root range: offset: " << kp.first << ", "
                    << "length: " << kp.second << "\n";

  auto *ht = new typename CellType::CellHashTable();
  auto *root = new CellType(r, 0, num_bodies, root_key, ht, b, attrs);
  ht->insert(std::make_pair(root_key, root));
  Refine(root, hn, b, 0, 0);

  return root;
#endif
}

template <class TSP>
void Partitioner<TSP>::Refine(Cell<TSP> *c,
                              const std::vector<HelperNode<TSP::Dim>> &hn,
                              const typename TSP::BT::type *b,
                              int cur_depth,
                              KeyType cur_key) const {
    const int Dim = TSP::Dim;
    typedef typename TSP::FP FP;
    typedef typename TSP::BT BT;

    TAPAS_LOG_INFO() << "Current depth: " << cur_depth << std::endl;
    if (c->nb() <= max_nb_) {
        TAPAS_LOG_INFO() << "Small enough cell" << std::endl;
        return;
    }
    if (cur_depth >= MAX_DEPTH) {
        TAPAS_LOG_INFO() << "Reached maximum depth" << std::endl;
        return;
    }
    KeyType child_key = MortonKeyFirstChild<Dim>(cur_key);
    index_t cur_offset = c->bid();
    index_t cur_len = c->nb();
    for (int i = 0; i < (1 << Dim); ++i) {
        TAPAS_LOG_DEBUG() << "Child key: " << child_key << std::endl;
        KeyPair kp = GetBodyRange<Dim, HelperNode<Dim>>(child_key,
                                                        hn.begin() + cur_offset,
                                                        hn.begin() + cur_offset + cur_len,
                                                        [](const HelperNode<Dim> &hn) { return hn.key; });
        index_t child_bn = kp.second;
        TAPAS_LOG_DEBUG() << "Range: offset: " << cur_offset << ", length: "
                          << child_bn << "\n";
        auto child_r = c->region().PartitionBSP(i);
        auto *child_cell = new Cell<TSP>(
            child_r, cur_offset, child_bn, child_key, c->ht(),
            c->bodies_, c->body_attrs_);
        c->ht()->insert(std::make_pair(child_key, child_cell));
        TAPAS_LOG_DEBUG() << "Particles: \n";
#ifdef TAPAS_DEBUG
        tapas::debug::PrintBodies<Dim, FP, BT>(b+cur_offset, child_bn, std::cerr);
#endif
        Refine(child_cell, hn, b, cur_depth+1, child_key);
        child_key = CalcMortonKeyNext<Dim>(child_key);
        cur_offset = cur_offset + child_bn;
        cur_len = cur_len - child_bn;
    }
    c->is_leaf_ = false;
}

} // namespace morton_hot

template <class TSP, class T2>
ProductIterator<CellIterator<morton_hot::Cell<TSP>>, T2>
Product(morton_hot::Cell<TSP> &c, T2 t2) {
    TAPAS_LOG_DEBUG() << "Cell-X product\n";
    typedef morton_hot::Cell<TSP> CellType;
    typedef CellIterator<CellType> CellIterType;
    return ProductIterator<CellIterType, T2>(CellIterType(c), t2);
}

template <class T1, class TSP>
ProductIterator<T1, CellIterator<morton_hot::Cell<TSP>>>
                         Product(T1 t1, morton_hot::Cell<TSP> &c) {
    TAPAS_LOG_DEBUG() << "X-Cell product\n";
    typedef morton_hot::Cell<TSP> CellType;
    typedef CellIterator<CellType> CellIterType;
    return ProductIterator<T1, CellIterType>(t1, CellIterType(c));
}

/**
 * @brief Constructs a ProductIterator for dual tree traversal of two trees
 */
template <class TSP>
ProductIterator<CellIterator<morton_hot::Cell<TSP>>,
                CellIterator<morton_hot::Cell<TSP>>>
                         Product(morton_hot::Cell<TSP> &c1,
                                 morton_hot::Cell<TSP> &c2) {
    TAPAS_LOG_DEBUG() << "Cell-Cell product\n";
    typedef morton_hot::Cell<TSP> CellType;
    typedef CellIterator<CellType> CellIterType;
    return ProductIterator<CellIterType, CellIterType>(
        CellIterType(c1), CellIterType(c2));
}


/**
 * @brief A partitioning plugin class that provides Morton-curve based octree partitioning.
 */
struct MortonHOT {
};

/**
 * @brief Advance decleration of a dummy class to achieve template specialization.
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR,
          class PartitionAlgorithm>
class Tapas;

/**
 * @brief Specialization of Tapas for HOT (Morton HOT) algorithm
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR>
class Tapas<DIM, FP, BT, BT_ATTR, CELL_ATTR, MortonHOT> {
    typedef TapasStaticParams<DIM, FP, BT, BT_ATTR, CELL_ATTR> TSP; // Tapas static params
  public:
    typedef tapas::Region<TSP> Region;
    typedef morton_hot::Cell<TSP> Cell;
    typedef tapas::BodyIterator<Cell> BodyIterator;

    /**
     * @brief Partition and build an octree of the target space.
     * @param b Array of body of BT::type.
     */
    static Cell *Partition(typename BT::type *b,
                           index_t nb, const Region &r,
                           int max_nb) {
        morton_hot::Partitioner<TSP> part(max_nb);
        return part.Partition(b, nb, r);
    }
};

} // namespace tapas

#endif // TAPAS_MORTON_HOT_
