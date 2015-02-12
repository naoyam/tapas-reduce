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
 * @brief Find a mapping F :: X -> Y and returns F(x).
 *
 * (1) If |X| <  |Y| and bidir is false, F is injective.
 * (2) If |X| <  |Y| and bidir is true,  F is neither injective nor surjective
 * (3) If |X| == |Y|, F is bijective.
 * (4) If |X| >  |Y|, F is surjective.
 */
template <class T>
std::vector<T> GetMapping(const std::vector<T> &X,
                          const std::vector<T> &Y,
                          bool bidir, // bidirectional
                          const T& x) {
  if (X.size() == 0 || Y.size() == 0) {
    return std::vector<T>();
  }
  
  int i = std::find(std::begin(X), std::end(X), x) - std::begin(X);
  assert(0 <= i && i < X.size());

  int a = X.size();
  int b = Y.size();
  
  if (a == b) {
    return std::vector<T>(1, Y[i]);
  } else if (a > b) {
    int q = a / b;
    int m = a % b;
    return std::vector<T>(1, Y[i / (q + m)]);
  } else { // |X| < |Y|
    if (bidir) {
      int q = b / a;
      int m = b % a;
      int beg = q * i + std::min<int>(m, i);
      int num = q + (i < m ? 1 : 0);
      int end = beg + num;
      return std::vector<T>(std::begin(Y) + beg,
                            std::begin(Y) + end);
    } else {
      return std::vector<T>(1, Y[i]);
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
    usleep(10000);
    MPI_Barrier(MPI_COMM_WORLD);
  }
}

template<class T1, class T2>
static void Dump(const T1 &bodies, const T2 &keys, std::ostream & strm) {
  for (int i = 0; i < bodies.size(); i++) {
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
  typedef typename TSP::BT::type BodyType;
  typedef typename TSP::BT_ATTR BodyAttrType;

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
   * @param leaf_owners Process ID (i.e. rank if in MPI) that cells belong to
   * @param local_bodies Array of bodies that the local process owns.
   * @param local_body_keys Keys of local bodies
   * @param local_body_attrs Array of BodyAttrType corresponding to local_body_keys
   */
  Cell(KeyType key,
       bool is_local, bool is_leaf,
       index_t body_beg, index_t body_num,
       std::shared_ptr<CellHashTable> ht,
       const Region<TSP> &region,
       VecPtr<KeyType>  leaf_keys,       // Keys of all (local and remote) leaf cells
       VecPtr<int>      leaf_owners,     // Process IDs that own i-th leaf cell.
       VecPtr<BodyType> local_bodies,    // Bodies which this process owns
       VecPtr<KeyType>  local_body_keys, // Keys of local_bodies
       VecPtr<BodyAttrType> local_body_attrs
       ) :
      tapas::BasicCell<TSP>(CalcRegion(key, region), body_beg, body_num),
      key_(key), is_local_(is_local), is_leaf_(is_leaf),
      ht_(ht),
      leaf_keys_(leaf_keys),
      leaf_owners_(leaf_owners),
      local_bodies_(local_bodies),
      local_body_keys_(local_body_keys),
      local_body_attrs_(local_body_attrs),
      owners_()
  {
#if 0

    // to be deleted
    std::cerr << "Cell(): key=" << key     << ", "
              << "is_leaf = " << is_leaf   << ", "
              << "nb = " << body_num       << ", "
              << "is_local = " << is_local << ", "
              << std::endl;
#endif
    CalcOwnerProcesses();
  }

  typedef typename TSP::ATTR attr_type;
  typedef typename TSP::BT_ATTR body_attr_type;
  KeyType key() const { return key_; }

  bool operator==(const Cell &c) const;
  template <class T>
  bool operator==(const T &) const { return false; }
  bool IsRoot() const;

  static void Map(Cell<TSP> &cell, std::function<void(Cell<TSP>&)> f);

  /**
   * @brief Returns if the cell is a leaf cell
   */
  bool IsLeaf() const;

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
  
  std::shared_ptr<CellHashTable> ht_; //!< Hash table of KeyType -> Cell* (only local cells)

  VecPtr<KeyType>  leaf_keys_;
  VecPtr<int>      leaf_owners_;
  VecPtr<BodyType> local_bodies_;
  VecPtr<KeyType>  local_body_keys_;
  VecPtr<BodyAttrType> local_body_attrs_;
  std::vector<int> owners_; //!< Processes that have this cell locally.
  
  // utility/accessor functions
  Cell *Lookup(KeyType k) const;
  CellHashTable *ht() { return ht_; }
  typename TSP::BT_ATTR &body_attr(index_t idx) const;
  virtual void make_pure_virtual() const {}

  static std::vector<int> CalcOwnerProcsOfCell(KeyType key,
                                               const std::vector<KeyType> &leaf_keys,
                                               const std::vector<int> &leaf_owners);
  void CalcOwnerProcesses();
  static Region<TSP> CalcRegion(KeyType, const Region<TSP>& r);

  // Map-related stuff
  void RecvCell(int pid);
  void SendCell(const std::vector<int> &pids);

  int GetMpiTag() const {
    return key_ % std::numeric_limits<int>::max();
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
  MPI_Allreduce(&r.max()[0], &new_max[0], Dim, MPI_DatatypeTraits<FP>::type(), MPI_MAX, MPI_COMM_WORLD);

  // Exchange min
  MPI_Allreduce(&r.min()[0], &new_min[0], Dim, MPI_DatatypeTraits<FP>::type(), MPI_MIN, MPI_COMM_WORLD);

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
  int rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  if (cell.is_local_) {
    f(cell);

    // Processes that needs the cell information can be identified by calculating the difference of sets
    //   {parent's owners} - {it's owners}
    
    if (cell.key() != 0) { // if not root
      // Owner processes of the remote cell can be obtained by
      // by {parent's owners} - {the cell's owners}
      std::vector<int> senders = cell.owners(); // Owner of the cells
      std::vector<int> recvers = SetDiff(cell.parent().owners(), senders); // Requesters
      std::vector<int> peers = GetMapping(senders, recvers, false, rank);
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
    
    int peer = GetMapping(recvers, senders, false, rank)[0];
    cell.RecvCell(peer); // the sender finishes applying f first, then send the cell info to this process.
  }
}

// A special-purpose struct to exchange cell information
template <class TSP>
struct CellInfoBinder {
  typename TSP::ATTR cell_attr;
  index_t num_bodies;
};

template <class TSP>
void Cell<TSP>::RecvCell(int pid) {
  // Send a recv request Cell/BodyAttr info to the process<pid>.
  //   Cell attributes
  //   Bodies (if the cell is a leaf)
  //   Body Attributes (if the cell is a leaf)

  int tag = this->GetMpiTag();

  // First, request CellAttr data. On sender side (in SendCell)
  CellInfoBinder<TSP> binder;

#ifdef DEBUG_SENDRECV
  {
    Stderr stderr("recv");
    stderr.out() << "RecvCell: cell=" << key_ << " "
                 << "src=" << pid << " "
                 << "tag=" << tag
                 << std::endl;
  }
#endif
  
  // note that this MPI_recv() call is handled by MPI_Isend in SendCell() in proc pid.
  MPI_Status stat;
#if 1
  MPI_Recv(&binder, sizeof(binder), MPI_BYTE, pid, tag, MPI_COMM_WORLD, &stat);
#else
  MPI_Request req;
  int flag = 0;
  MPI_Irecv(&binder, sizeof(binder), MPI_BYTE, pid, tag, MPI_COMM_WORLD, &req);
  while(!flag) {
    MPI_Test(&req, &flag, &stat);
    usleep(1000);
  }
#endif
  
  //assert(stat.MPI_ERROR == MPI_SUCCESS);
  this->attr() = binder.cell_attr;
  this->nb_ = binder.num_bodies;
  
#ifdef DEBUG_SENDRECV
  {
    Stderr stderr("recv");
    stderr.out() << "RecvCell: cell=" << key_ << " "
                 << "src=" << pid << " "
                 << "tag=" << tag << " "
                 << "done" << std::endl;
  }
#endif
}

template <class TSP>
void Cell<TSP>::SendCell(const std::vector<int> &pids) {
  if (pids.size() == 0) {
    return;
  }

  int tag = this->GetMpiTag();

  CellInfoBinder<TSP> binder {this->attr(), this->nb()};
  MPI_Status *stats = new MPI_Status[pids.size()];
  MPI_Request *reqs = new MPI_Request[pids.size()];

  for (int i = 0; i < pids.size(); i++) {
#ifdef DEBUG_SENDRECV
    {
      Stderr stderr("send");
      stderr.out() << "SendCell: cell=" << key_ << " "
                   << "dst=" << pids[i] << " "
                   << "tag=" << tag
                   << std::endl;
    }
#endif
    MPI_Isend(&binder, sizeof(binder), MPI_BYTE, pids[i], tag, MPI_COMM_WORLD, &reqs[i]);
  }

  int flag = 0;
  while(!flag) {
    MPI_Testall(pids.size(), reqs, &flag, stats);
    usleep(100);
  }

  delete[] stats;
  delete[] reqs;
  
#ifdef DEBUG_SENDRECV
  for (int i = 0; i < pids.size(); i++) {
    Stderr stderr("send");
    stderr.out() << "SendCell: cell=" << key_ << " "
                 << "dst=" << pids[i] << " "
                 << "tag=" << tag << " "
                 << "done" << std::endl;
  }
#endif
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

  KeyType k = MortonKeyChild<TSP::Dim>(key_, idx);
  Cell *c = Lookup(k);

  if (c == NULL) {
    // This means c is a remote cell (owned by other processes)
    // and this is the first access to the remote cell.
    bool is_leaf = find(leaf_keys_->begin(), leaf_keys_->end(), k) == leaf_keys_->end();
    c = new Cell(k, false, is_leaf, 0, 0, ht_, this->region(),
                 leaf_keys_, leaf_owners_, local_bodies_, local_body_keys_, local_body_attrs_);
    (*ht_)[k] = c;
  }
  return *Lookup(k);
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
                          << "cell (" << key_ << ") not found."
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
                                         const std::vector<long> &nb,
                                         int max_nb) {
  std::vector<KeyType> ret; // new

  for (int i = 0; i < cell_keys.size(); i++) {
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
inline std::vector<int> SplitKeysSimple(const std::vector<long> &nb, int proc_size) {
  index_t total_nb = std::accumulate(nb.begin(), nb.end(), 0); // total number of bodies
  index_t guide = total_nb / proc_size;

  std::vector<int> ret(nb.size()); // return value
  int psum = 0; // partial sum
  int cur_proc = 0;

  for (int i = 0; i < nb.size(); i++) {
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

template <class TSP>
Region<TSP> Cell<TSP>::CalcRegion(KeyType key, const Region<TSP> &region) {
  const int kDim = TSP::Dim;
  const int kMask = (1 << TSP::Dim) - 1;

  Stderr err("center");

  auto r = region;
  int depth = MortonKeyGetDepth(key);
  KeyType key_body = MortonKeyRemoveDepth(key);
  
  err.out() << std::setw(20) << key << " "
            << depth << " "
            << r << " ";

  for (int d = 0; d < depth; d++) {
    int direction = (key_body >> (kDim * (MAX_DEPTH - d - 1))) & kMask;
    r = r.PartitionBSP(direction);
    key >>= TSP::Dim;
    err.out() << d << ":" << direction << " " << r << " ";
  }
  
  err.out() << std::endl;
  
  return r;
}

template <class TSP>
std::vector<int> Cell<TSP>::CalcOwnerProcsOfCell(KeyType key,
                                                 const std::vector<KeyType> &leaf_keys,
                                                 const std::vector<int> &leaf_owners) {
  // Find the range of leaf cells that are under the cell
  auto beg = std::begin(leaf_keys);
  auto end = std::end(leaf_keys);
  KeyType key_next = morton_common::CalcMortonKeyNext<TSP::Dim>(key);
  size_t owner_beg = std::lower_bound(beg, end, key) - beg;
  size_t owner_end = std::lower_bound(beg, end, key_next) - beg;
  
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

  BarrierExec([&r](int rank, int size) {
      std::cerr << "Region: rank " << rank << ": " << r << std::endl;
    });

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
  std::vector<long> leaf_nb_local;  // Number of local bodies in leaf cell[i] (all global cells)
  std::vector<long> leaf_nb_global; // Number of global bodies in leaf cell[i] (all global cells)

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

#if 0
    //--------------------------------------------------------------
    // debug print (to be deleted)
    MPI_Barrier(MPI_COMM_WORLD);
    std::cout.flush(); std::cerr.flush();
    usleep(10000);

    const int w = 3;
    if (rank == 0) {
      std::cerr << "-------------------------------------------" << std::endl;
      std::cerr << "Partition() MAX_DEPTH_BY_DEPTH_BITS = " << MAX_DEPTH_BY_DEPTH_BITS << std::endl;
      std::cerr << "Partition() MAX_DEPTH_BY_KEY_BITS = " << MAX_DEPTH_BY_KEY_BITS << std::endl;
      std::cerr << "Partition() MAX_DEPTH = " << MAX_DEPTH << std::endl;
      std::cerr << "CalcMortonKeyNext(0) = " << CalcMortonKeyNext<Dim>(0) << std::endl;

      KeyType test_key = 70368744177665UL;
      KeyType first_child = MortonKeyFirstChild<Dim>(test_key);
      KeyType inc_depth = morton_common::MortonKeyIncrementDepth(test_key, 1);
      std::cerr << "MortonKeyFirstChild test." << std::endl;
      std::cerr << "Parent : " << std::endl;
      std::cerr << std::setw(15) << std::right << test_key << " " << MortonKeyDecode<Dim>(test_key) << std::endl;
      std::cerr << "First child : " << std::endl;
      std::cerr << std::setw(15) << std::right << first_child << " " << MortonKeyDecode<Dim>(first_child) << std::endl;
      std::cerr << "inc depth : " << std::endl;
      std::cerr << std::setw(15) << std::right << inc_depth << " " << MortonKeyDecode<Dim>(inc_depth) << std::endl;

      std::cerr << "MPI size = " << size << std::endl;
      std::cerr << std::left << std::fixed << std::setw(10) << "index";
      for (int i = 0; i < leaf_keys.size(); i++) {
        std::cerr << std::fixed << std::setw(w) << i << " ";
      }
      std::cerr << std::endl;

      std::cerr << std::left << std::fixed << std::setw(10) << "depths";
      for (auto k : leaf_keys) {
        std::cerr << std::fixed << std::setw(w) << MortonKeyGetDepth(k) << " ";
      }
      std::cerr << std::endl;

      std::cerr << std::left << std::fixed << std::setw(10) << "keys" << std::endl;;
      for (auto k : leaf_keys) {
        std::cerr << std::setw(15) << std::right << k << " (" << MortonKeyDecode<Dim>(k) << ")" << std::endl;
      }
      std::cerr << std::endl;

      std::cerr << std::left << std::fixed << std::setw(10) << "nb_gl";
      for (auto nb : leaf_nb_global) {
        std::cerr << std::fixed << std::setw(w) << nb << " ";
      }
      std::cerr << std::endl;
    }

    BarrierExec([&] (int rank, int size) {
        std::cerr << "rank " << std::fixed << std::setw(3) << std::left << rank << "  ";
        for (auto nb : leaf_nb_local) {
          std::cerr << std::fixed << std::setw(w) << nb << " ";
        }
        std::cerr << std::endl;
      });
    // debug print ends
    //--------------------------------------------------------------
#endif

    if (rank == 0) {
      constexpr int w = 9;
      auto max_depth_func = [](int d, KeyType k) {
        return std::max(d, MortonKeyGetDepth(k));
      };
      const int max_depth = std::accumulate(leaf_keys.begin(),
                                            leaf_keys.end(),
                                            0, max_depth_func);
      std::cerr << "max_nb = " << std::setw(w) << max_nb << " "
                << "#cells = " << std::setw(w) << leaf_keys.size() << " "
                << "maxdepth = " << std::setw(w) << max_depth << " "
                << "total = " << sum(leaf_nb_global)
                << std::endl;
    }

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

  if (rank == 0) {
#if 0
    std::cerr << "Cells are: " << std::endl;
    for (int i = 0; i < leaf_nb_global.size(); i++) {
      std::cerr << std::fixed << std::setw(3) << i << " ";
    }
    std::cerr << std::endl;
    for (int i = 0; i < leaf_nb_global.size(); i++) {
      std::cerr << std::fixed << std::setw(3) << leaf_nb_global[i] << " ";
    }
    std::cerr << std::endl;
    for (int i = 0; i < leaf_nb_global.size(); i++) {
      std::cerr << std::fixed << std::setw(3) << leaf_owners[i] << " ";
    }
    std::cerr << std::endl;
#endif
    
    std::cerr << "Number of cells each process owns" << std::endl;
    for (int pi = 0; pi < size; pi++) {
      auto beg = std::lower_bound(leaf_owners.begin(), leaf_owners.end(), pi);
      auto end = std::upper_bound(leaf_owners.begin(), leaf_owners.end(), pi);
      int ncells = std::count(leaf_owners.begin(), leaf_owners.end(), pi);
      std::cerr << std::fixed << std::setw(3) << pi << " "
                << std::fixed << std::setw(3) << ncells
                << "(from " << (beg - leaf_owners.begin()) << ", "
                << (end - leaf_owners.begin()) << ")"
                << std::endl;
    }
  }

  // Exchange bodies using MPI_Alltoallv
  std::vector<int> send_counts(size, 0); // number of bodies that this process sends to others
  std::vector<int> recv_counts(size, 0); // number of bodies that this process receives from others

  BarrierExec([&](int rank, int size) {
      std::cerr << "Rank " << rank << " "
                << "leaf_keys.size() = " << leaf_keys.size() << " "
                << "leaf_owners.size() = " << leaf_owners.size() << " "
                << std::endl;
    });

  for (int ci = 0; ci < leaf_keys.size(); ci++) {
    // count bodies to be sent to process 'proc' in cell ci
    // note that send_count and recv_count are multiplied by sizeof(BodyType) so that
    // BodyType objects will be sent as arrays of MPI_BYTE
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
  for (int hi = 0; hi < hn.size(); hi++) {
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
  for (int pi = 0; pi < recv_counts.size(); pi++) {
    if (pi == 0) {
      recv_disp_bodies[pi] = 0;
      recv_disp_keys[pi] = 0;
    } else {
      recv_disp_bodies[pi] = recv_disp_bodies[pi-1] + recv_bytes_bodies[pi-1];
      recv_disp_keys[pi] = recv_disp_keys[pi-1] + recv_bytes_keys[pi-1];
    }
  }

#if 0
  // Debug message of BodyExchange phase
  if (rank == 0) {
    std::cerr << "---------------------------" << std::endl;
    std::cerr << "(sizeof(BodyType) = " << sizeof(BodyType) << ")" << std::endl;
  }

  BarrierExec([&](int rank, int size) {
      for (int p = 0; p < size; p++) {
        std::cerr << "Rank "
                  << std::fixed << std::setw(3) << rank
                  << " sends "
                  << std::fixed << std::setw(6) << send_bytes_bodies[p] << " bytes "
                  << "(" << (send_counts[p]) << " bodies)"
                  << " to " << p
                  << std::endl;
      }

      std::cerr << "Rank "
                << std::fixed << std::setw(3) << rank << " " << "send_disp_bodies = ";
      for (int i = 0; i < send_disp_bodies.size(); i++) {
        std::cerr << std::fixed << std::setw(4) << send_disp_bodies[i]
                  << "(" << send_disp_bodies[i] / sizeof(BodyType) << " bodies)"
                  << " ";
      }
      std::cerr << std::endl;

      for (int p = 0; p < size; p++) {
        std::cerr << "Rank "
                  << std::fixed << std::setw(3) << rank
                  << " recvs "
                  << std::fixed << std::setw(6) << recv_bytes_bodies[p] << " bytes "
                  << "(" << recv_counts[p] << " bodies)"
                  << " from " << p
                  << std::endl;
      }

      std::cerr << "Rank "
                << std::fixed << std::setw(3) << rank << " " << "recv_disp_bodies = ";
      for (int i = 0; i < recv_disp_bodies.size(); i++) {
        std::cerr << std::fixed << std::setw(4) << recv_disp_bodies[i]
                  << "(" << recv_disp_bodies[i] / sizeof(BodyType) << " bodies)"
                  << " ";
      }
      std::cerr << std::endl;

      std::cerr << "Rank "
                << std::fixed << std::setw(3) << rank << " "
                << std::accumulate(recv_bytes_bodies.begin(), recv_bytes_bodies.end(), 0)
                << " bytes, "
                << std::accumulate(recv_counts.begin(), recv_counts.end(), 0) / sizeof(BodyType)
                << " bodies total"
                << std::endl;
    });
#endif

  // Call MPI_Alltoallv() for bodies
  MPI_Alltoallv(send_bodies.data(),  send_bytes_bodies.data(), send_disp_bodies.data(), MPI_BYTE,
                recv_bodies->data(), recv_bytes_bodies.data(), recv_disp_bodies.data(), MPI_BYTE,
                MPI_COMM_WORLD);

  // Call MPI_Alltoallv() for body keys
  MPI_Alltoallv(send_keys.data(), send_bytes_keys.data(), send_disp_keys.data(), MPI_BYTE,
                recv_keys.data(), recv_bytes_keys.data(), recv_disp_keys.data(), MPI_BYTE,
                MPI_COMM_WORLD);

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

  // construct a local tree from cells which belong to this process.
  auto leaf_beg = std::lower_bound(std::begin(leaf_owners), std::end(leaf_owners), rank) - std::begin(leaf_owners);
  auto leaf_end = std::upper_bound(std::begin(leaf_owners), std::end(leaf_owners), rank) - std::begin(leaf_owners);

  // heap copy of vectors (shared by Cells)
  auto leaf_owners2     = std::make_shared<std::vector<int>>(leaf_owners);
  auto leaf_keys2       = std::make_shared<std::vector<KeyType>>(leaf_keys);
  auto local_body_keys  = std::make_shared<std::vector<KeyType>>(recv_keys);
  auto local_body_attrs = std::make_shared<std::vector<BodyAttrType>>(num_bodies_recv);

  std::vector<Cell<TSP>*> interior_cells;

  // Build a local tree in a bottom-up manner.
  for (auto i = leaf_beg; i < leaf_end; i++) {
    KeyType k = leaf_keys[i];
    KeyType kn = CalcMortonKeyNext<Dim>(k);

    // bodies that the Cell, of which key is k, owns.
    auto bbeg = std::lower_bound(std::begin(recv_keys), std::end(recv_keys), k ) - std::begin(recv_keys);
    auto bend = std::lower_bound(std::begin(recv_keys), std::end(recv_keys), kn) - std::begin(recv_keys);
    //auto bend = std::upper_bound(std::begin(recv_keys), std::end(recv_keys), kn) - std::begin(recv_keys);

    // Create a leaf cell
    CellType *c = new CellType(k,               // key
                               true,            // is_local
                               true,            // is_leaf
                               bbeg,            // body index
                               bend - bbeg,     // #bodies
                               ht,              // CellHashTable
                               r,               // Region region
                               leaf_keys2,
                               leaf_owners2,
                               recv_bodies,     // local bodies
                               local_body_keys, // local body keys
                               local_body_attrs);     // body attrs
    (*ht)[k] = c;
    assert(c->IsLocal() && c->IsLeaf());

    // Create anscestors of the cell c
    while(1) {
      k = MortonKeyParent<Dim>(k);
      kn = CalcMortonKeyNext<Dim>(k);
      int dp = MortonKeyGetDepth(k);
      int bbeg = std::lower_bound(std::begin(recv_keys), std::end(recv_keys), k ) - std::begin(recv_keys);
      int bend = std::lower_bound(std::begin(recv_keys), std::end(recv_keys), kn) - std::begin(recv_keys);

      if (ht->count(k) == 0) {
        // Create interior cellls (anscestors)
        CellType *c = new CellType(k,                 // key
                                   true,              // is_local
                                   false,             // is_leaf
                                   bbeg, bend - bbeg,
                                   ht,                // CellHashTable
                                   r,                 // Region region
                                   leaf_keys2,
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
  // Created all local cells

  // Create remote cells (proxy cells)
  for (auto &&c : interior_cells) {
    int nchld = c->nsubcells();
    for (int i = 0; i < nchld; i++) {
      c->subcell(i);
    }
  }
  

  // Dump all (local) cells to a file
  {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    std::stringstream ss;
    ss << "Cells." << rank << ".dat";
    std::ofstream ofs(ss.str().c_str());
    for (auto&& iter : (*ht)) {
      KeyType k = iter.first;
      Cell<TSP> *c = iter.second;
      ofs << std::setw(20) << std::right << k << " "
          << MortonKeyGetDepth(k) << " "
          << "[" << c->center() << "]"
          << std::endl;
    }
    ofs.close();
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
