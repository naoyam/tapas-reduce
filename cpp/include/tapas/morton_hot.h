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
#include <numeric>
#include <list>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <iomanip>

#include <unistd.h>
#include <mpi.h>

#include "tapas/cell.h"
#include "tapas/bitarith.h"
#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/iterator.h"
#include "tapas/morton_common.h"

namespace tapas {

/**
 * @brief Provides MPI-based distributed Morton-order octree partitioning
 */
namespace morton_hot {

using namespace morton_common;

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

template <class TSP>
std::vector<HelperNode<TSP::Dim>>
CreateInitialNodes(const typename TSP::BT::type *p, index_t np, 
                   const Region<TSP> &r);

KeyType MortonKeyAppendDepth(KeyType k, int depth);

KeyType MortonKeyRemoveDepth(KeyType k);
    
KeyType MortonKeyIncrementDepth(KeyType k, int inc);

template <int DIM>
KeyType MortonKeyClearDescendants(KeyType k);

template <int DIM>
KeyType MortonKeyParent(KeyType k);

template <int DIM>
KeyType MortonKeyFirstChild(KeyType k);

template <int DIM>
KeyType MortonKeyChild(KeyType k, int child_idx);


template <int DIM, class T>
void AppendChildren(KeyType k, T &s);

template <int DIM, class BT>
void SortBodies(const typename BT::type *b, typename BT::type *sorted,
                const HelperNode<DIM> *nodes,
                tapas::index_t nb);

template <int DIM>
KeyType FindFinestAncestor(KeyType x, KeyType y);

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
    typedef unordered_map<KeyType, Cell*> HashTable;
  protected:
    KeyType key_;
    HashTable *ht_;
  public:
    Cell(const Region<TSP> &region,
         index_t bid, index_t nb, KeyType key,
         HashTable *ht,
         typename TSP::BT::type *bodies,
         typename TSP::BT_ATTR *body_attrs) :
            tapas::BasicCell<TSP>(region, bid, nb), key_(key),
            ht_(ht), bodies_(bodies), body_attrs_(body_attrs),
            is_leaf_(true) {}
    
    typedef typename TSP::ATTR attr_type;
    typedef typename TSP::BT_ATTR body_attr_type;
    KeyType key() const { return key_; }

    bool operator==(const Cell &c) const;
    template <class T>
    bool operator==(const T &) const { return false; }
    bool IsRoot() const;
    bool IsLeaf() const;
    int nsubcells() const;
    Cell &subcell(int idx) const;
    Cell &parent() const;
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
    typename TSP::BT_ATTR &body_attr(index_t idx) const;
    HashTable *ht() { return ht_; }
    Cell *Lookup(KeyType k) const;
    typename TSP::BT::type *bodies_;
    typename TSP::BT_ATTR *body_attrs_;
    bool is_leaf_;
    virtual void make_pure_virtual() const {}
}; // class Cell


/**
 * @brief Set depth information in a Morton key.
 */
inline
KeyType MortonKeyAppendDepth(KeyType k, int depth) {
  return (k << DEPTH_BIT_WIDTH) | depth;
}

inline
KeyType MortonKeyRemoveDepth(KeyType k) {
    return k >> DEPTH_BIT_WIDTH;
}

inline
KeyType MortonKeyIncrementDepth(KeyType k, int inc) {
    int depth = MortonKeyGetDepth(k);
    depth += inc;
#ifdef TAPAS_DEBUG
    if (depth > MAX_DEPTH) {
        TAPAS_LOG_ERROR() << "Exceeded the maximum allowable depth: " << MAX_DEPTH << std::endl;
        TAPAS_DIE();
    }
#endif  
    k = MortonKeyRemoveDepth(k);
    return MortonKeyAppendDepth(k, depth);
}


// MPI-related utilities and wrappers
// TODO: wrap them as a pluggable policy class 
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
                                                     const Region<TSP> &reg) {
    const int Dim = TSP::Dim;
    typedef typename TSP::FP FP;
    typedef typename TSP::BT BT;

    auto r = ExchangeRegion(reg);

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

template <int DIM>
KeyType FindFinestAncestor(KeyType x,
                                  KeyType y) {
    int min_depth = std::min(MortonKeyGetDepth(x),
                             MortonKeyGetDepth(y));
    x = MortonKeyRemoveDepth(x);
    y = MortonKeyRemoveDepth(y);
    KeyType a = ~(x ^ y);
    int common_depth = 0;
    for (; common_depth < min_depth; ++common_depth) {
        KeyType t = (a >> (MAX_DEPTH - common_depth -1) * DIM) & ((1 << DIM) - 1);
        if (t != ((1 << DIM) -1)) break;
    }
    int common_bit_len = common_depth * DIM;
    KeyType mask = ((1 << common_bit_len) - 1) << (MAX_DEPTH * DIM - common_bit_len);
    return MortonKeyAppendDepth(x & mask, common_depth);
}

template <int DIM, class T>
void AppendChildren(KeyType x, T &s) {
    int x_depth = MortonKeyGetDepth(x);
    int c_depth = x_depth + 1;
    if (c_depth > MAX_DEPTH) return;
    x = MortonKeyIncrementDepth(x, 1);
    for (int i = 0; i < (1 << DIM); ++i) {
        int child_key = (i << ((MAX_DEPTH - c_depth) * DIM + DEPTH_BIT_WIDTH));
        s.push_back(x | child_key);
        TAPAS_LOG_DEBUG() << "Adding child " << (x | child_key) << std::endl;
    }
}

template <int DIM>
KeyType MortonKeyClearDescendants(KeyType k) {
    int d = MortonKeyGetDepth(k);
    KeyType m = ~(((1 << ((MAX_DEPTH - d) * DIM)) - 1) << DEPTH_BIT_WIDTH);
    return k & m;
}

template <int DIM>
KeyType MortonKeyParent(KeyType k) {
    int d = MortonKeyGetDepth(k);  
    if (d == 0) return k;
    k = MortonKeyIncrementDepth(k, -1);
    return MortonKeyClearDescendants<DIM>(k);
}


template <int DIM>
KeyType MortonKeyFirstChild(KeyType k) {
#ifdef TAPAS_DEBUG
    KeyType t = MortonKeyRemoveDepth(k);
    t = t & ~(~((KeyType)0) << (DIM * (MAX_DEPTH - MortonKeyGetDepth(k))));
    assert(t == 0);
#endif
    assert(MortonKeyGetDepth(k) < MAX_DEPTH);
    return MortonKeyIncrementDepth(k, 1);
}

template <int DIM>
KeyType MortonKeyChild(KeyType k, int child_idx) {
    TAPAS_ASSERT(child_idx < (1 << DIM));
    k = MortonKeyIncrementDepth(k, 1);
    int d = MortonKeyGetDepth(k);
    return k | (child_idx << ((MAX_DEPTH - d) * DIM + DEPTH_BIT_WIDTH));
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
int Cell<TSP>::nsubcells() const {
    if (IsLeaf()) return 0;
    else return (1 << TSP::Dim);
}

template <class TSP>
Cell<TSP> &Cell<TSP>::subcell(int idx) const {
    KeyType k = MortonKeyChild<TSP::Dim>(key_, idx);
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
    return bodies_[this->bid_+idx];
}

template <class TSP>
typename TSP::BT_ATTR *Cell<TSP>::body_attrs() const {
    return body_attrs_;
}

template <class TSP>
typename TSP::BT_ATTR &Cell<TSP>::body_attr(index_t idx) const {
    return body_attrs_[this->bid_+idx];
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
 * @brief Split cells that have more than nb_max bodies
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
      
#if 0
      std::cerr << "Splitting " << std::endl;
      std::cerr << std::setw(15) << std::right << cell_keys[i] << " (" << MortonKeyDecode<DIM>(cell_keys[i]) << ")" << std::endl;
      std::cerr << "Children are:" << std::endl;
      for (auto ch : children) {
        std::cerr << std::setw(15) << std::right << ch << " (" << MortonKeyDecode<DIM>(ch) << ")" << std::endl;
      }
#endif
      
    }
  }

  return ret;
}

/**
 * @brief Distribute cells over processes in a simple algorithm so that each process has roughly equal number of bodies
 * @param nb Array of numbers of bodies in cells
 * @param proc_size Number of processes (assumes process numbers are integer starting from 0)
 * @return Array of process numbers 
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
      cur_proc++;
    }
  }

  return ret;
}

/**
 * @brief Partition the simulation space and build Morton-key based octree
 * @tparam TSP Tapas static params
 * @param b Array of particles
 * @param numBodies Length of b (NOT the total number of bodies over all processes)
 * @param r Geometry of the target space
 * @return The root cell of the constructed tree
 */
template <class TSP> // TSP : Tapas Static Params
Cell<TSP>*
Partitioner<TSP>::Partition(typename TSP::BT::type *b,
                            index_t num_bodies,
                            const Region<TSP> &r) {
  const int Dim = TSP::Dim;
  typedef typename TSP::FP FP;
  typedef typename TSP::BT BT;
  typedef typename TSP::BT_ATTR BodyAttrType;
  typedef typename BT::type BodyType;
  typedef Cell<TSP> CellType;
  typedef HelperNode<Dim> HN;

  // Sort local bodies using Morton keys
  std::vector<HN> hn = CreateInitialNodes<TSP>(b, num_bodies, r);
  std::sort(hn.begin(), hn.end(),
            [](const HN &lhs, const HN &rhs) { return lhs.key < rhs.key; });
    
  std::vector<long> leaf_nb_local;  // Number of local bodies in leaf cell[i]
  std::vector<long> leaf_nb_global; // Number of global bodies in leaf cell[i]
  std::vector<KeyType> leaf_keys;      // Morton keys of leaf cells

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
      std::cerr << "MortonKeyFirstChild test." << std::endl;
      std::cerr << "Parent : " << std::endl;
      std::cerr << std::setw(15) << std::right << test_key << " " << MortonKeyDecode<Dim>(test_key) << std::endl;
      std::cerr << "First child : " << std::endl;
      std::cerr << std::setw(15) << std::right << first_child << " " << MortonKeyDecode<Dim>(first_child) << std::endl;
        
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
                << "total = " << std::accumulate(leaf_nb_global.begin(), leaf_nb_global.end(), 0)
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
      int ncells = std::count(leaf_owners.begin(), leaf_owners.end(), pi);
      std::cerr << std::fixed << std::setw(3) << pi << " "
                << std::fixed << std::setw(3) << ncells << std::endl;
    }
  }

  MPI_Finalize();
  exit(0);

  //-------------
    
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

  auto *ht = new typename CellType::HashTable();
  auto *root = new CellType(r, 0, num_bodies, root_key, ht, b, attrs);
  ht->insert(std::make_pair(root_key, root));
  Refine(root, hn, b, 0, 0);
    
  return root;
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
