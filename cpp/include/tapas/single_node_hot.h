/**
 * @file single_node_hot.h
 * @brief Implements single node SFC-based HOT (Hashed Octree) implementation
 */
#ifndef TAPAS_SINGLE_NODE_HOT_
#define TAPAS_SINGLE_NODE_HOT_

#include "tapas/stdcbug.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <list>
#include <vector>
#include <unordered_set>
#ifdef __CUDACC__
#include <unordered_map>
//#include <tr1/unordered_map>
#else
#include <unordered_map>
#endif
#include <utility>
#include <iostream>
#include <iomanip>
#include <functional>
#include <memory>

// for debugging
#include <fstream>

#include "tapas/cell.h"
#include "tapas/bitarith.h"
#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/iterator.h"
//#include "tapas/morton_common.h"
#include "tapas/sfc_morton.h"
#include "tapas/single_node_mapper.h"

namespace {
namespace iter = tapas::iterator;
}

namespace tapas {

/**
 * @brief Provides SFC-based octree partitioning for shared memory single node
 */
namespace single_node_hot {

template <class TSP>
struct HelperNode {
  typename TSP::SFC::KeyType key;          //!< SFC key
  Vec<TSP::Dim, int> anchor; //!< SFC-key like vector without depth information
  index_t p_index;      //!< Index of the corresponding body
  index_t np;           //!< Number of particles in a node
};

template <class TSP>
std::vector<HelperNode<TSP>>
CreateInitialNodes(const typename TSP::BT::type *p, index_t np, 
                   const Region<TSP> &r);

template <int DIM, class Key, class T>
void AppendChildren(typename Key::KeyType k, T &s);

template <class TSP>
void SortBodies(const typename TSP::BT::type *b, typename TSP::BT::type *sorted,
                const HelperNode<TSP> *nodes,
                tapas::index_t nb);

template <class TSP>
void CompleteRegion(typename TSP::SFC::KeyType x,
                    typename TSP::SFC::KeyType y,
                    typename TSP::SFC::KeyVector &s);

template <class TSP> class Partitioner;
template <class TSP> class Cell;

/**
 * Data shared between all cells
 */
template<class TSP>
class SharedData {
 public:
  using CellType = Cell<TSP>;
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;
#ifdef __CUDACC__
  using HashTable = std::unordered_map<KeyType, CellType*>;
#else
  using HashTable = std::unordered_map<KeyType, CellType*>;
#endif
  
  using BodyType = typename TSP::Body;
  using BodyAttrType = typename TSP::BodyAttr;

  HashTable ht_;
  bool opt_mutual_;

  SharedData() :
      ht_(),
      opt_mutual_(false)
  { }

  SharedData(const SharedData<TSP>& rhs) = delete; // no copy
  SharedData(SharedData<TSP>&& rhs) = delete; // no move
};

template <class TSP> // TapasStaticParams
class Cell: public tapas::BasicCell<TSP> { 
  friend class Partitioner<TSP>;
  friend class iter::BodyIterator<Cell>;
 public:
  typedef typename TSP::CellAttr attr_type;
  typedef typename TSP::Body BodyType;
  typedef typename TSP::BodyAttr BodyAttrType; // to be fixed
  typedef typename TSP::BodyAttr BodyAttr;
  typedef typename TSP::Threading Threading;
  
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;

  using HashTable = typename SharedData<TSP>::HashTable;
  using CellType = Cell<TSP>;
  using Body = typename TSP::Body;
  using BodyIterator = iter::BodyIterator<CellType>;
  using SubCellIterator = iter::SubCellIterator<CellType>;
  using Mapper = typename TSP::template Mapper<CellType, Body, NONE>;
  
 protected:
  std::shared_ptr<SharedData<TSP>> data_;
  KeyType key_;
  index_t nb_; //!< number of local bodies of the process.
  Mapper mapper_;
 public:
  Cell(std::shared_ptr<SharedData<TSP>> data,
       const Region<TSP> &region,
       index_t bid, index_t nb, KeyType key,
       typename TSP::Body *bodies,
       typename TSP::BodyAttr *body_attrs) :
      tapas::BasicCell<TSP>(region, bid, nb), data_(data), key_(key),
      nb_(nb), bodies_(bodies), body_attrs_(body_attrs),
      is_leaf_(true) {}

  inline CellType &cell() { return *this; }
  inline const CellType &cell() const { return *this; }

  static void PostOrderMap(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f);
  static void PreOrderMap(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f);
  
    KeyType key() const { return key_; }

  bool operator==(const Cell &c) const;
  bool operator<(const Cell &c) const;
    template <class T>
    bool operator==(const T &) const { return false; }
    bool IsRoot() const;
    bool IsLeaf() const;
    int nsubcells() const;
    size_t local_nb() const { return nb_; } 
    Cell &subcell(int idx) const;
    Cell &parent() const;

  bool IsLocal() const { return true; }

  INLINE Mapper& mapper() { return mapper_; }
  INLINE const Mapper& mapper() const { return mapper_; }
  
#ifdef DEPRECATED
  typename TSP::Body &particle(index_t idx) const {
    return body(idx);
  }
#endif

  // Accessor functions to bodies & body attributes
  
  /**
   * \brief returns idx-th body in local memory
   * In single node HOT, the set of bodies are equivalent to the bodies users first
   * gave to tapas, but they might be re-ordered.
   */
  BodyType& body(index_t idx);
  const BodyType& body(index_t idx) const;
  BodyType& local_body(index_t idx);
  const BodyType& local_body(index_t idx) const;

  /**
   * \brief Returns an attribute of the idx-th local body.
   */
  BodyAttrType &body_attr(index_t idx);
  const BodyAttrType &body_attr(index_t idx) const;
  
  BodyAttrType &local_body_attr(index_t idx) {
    return body_attrs_[this->bid_+idx];
  }

  const BodyAttrType &local_body_attr(index_t idx) const {
    return body_attrs_[this->bid_+idx];
  }
  
  BodyIterator bodies() {
    return BodyIterator(*this);
  }
  
  BodyAttrType *body_attrs();
  const BodyAttrType *body_attrs() const;
  
  BodyAttrType *local_body_attrs() {
    return body_attrs_;
  }
  
  const BodyAttrType *local_body_attrs() const {
    return body_attrs_;
  }
  
#ifdef DEPRECATED
  typename TSP::BodyAttr *particle_attrs() const {
    return body_attrs();
  }
#endif
  
  SubCellIterator subcells() {
    return SubCellIterator(*this);
  }

  int depth() const {
    return SFC::GetDepth(key_);
  }

  void Report() const {
  }

  bool GetOptMutual() const {
    return data_->opt_mutual_;
  }

  bool SetOptMutual(bool b) {
    bool prev = data_->opt_mutual_;
    data_->opt_mutual_ = b;
    return prev;
  }
  
 protected:
  HashTable &ht() { return data_->ht_; }
  const HashTable &ht() const { return data_->ht_; }
  
  Cell *Lookup(KeyType k) const;
  typename TSP::Body *bodies_;
  typename TSP::BodyAttr *body_attrs_;
  bool is_leaf_;
  
  const std::vector<BodyType>& LocalBodies() const;
  const std::vector<BodyAttrType>& LocalBodyAttrs() const;
}; // class Cell


/**
 * @brief Create an array of HelperNode from bodies
 * In the first stage of tree construction, one HelperNode is create for each body.
 * @return Array of HelperNode
 * @param p A pointer to an array of bodies
 * @param np Number of bodies
 * @param r Region object
 */
template <class TSP>
std::vector<HelperNode<TSP>> CreateInitialNodes(const typename TSP::Body *p,
                                                index_t np,
                                                const Region<TSP> &r) {
    const constexpr int Dim = TSP::Dim;
    const constexpr size_t kCoordOfst = TSP::kBodyCoordOffset;
    using SFC = typename TSP::SFC;
    //using KeyType = typename SFC::KeyType;
    using FP = typename TSP::FP;
    
    std::vector<HelperNode<TSP>> nodes(np);
    FP num_cell = 1 << SFC::MAX_DEPTH; // maximum number of cells in one dimension
    Vec<Dim, FP> pitch;           // possible minimum cell width
    for (int d = 0; d < Dim; ++d) {
        pitch[d] = (r.max()[d] - r.min()[d]) / num_cell;
    }

    for (index_t i = 0; i < np; ++i) {
        // First, create 1 helper cell per particle
        HelperNode<TSP> &node = nodes[i];
        node.p_index = i;
        node.np = 1;

        // Particle pos offset is the offset of each coordinate value (x,y,z) in body structure
        Vec<Dim, FP> off = ParticlePosOffset<Dim, FP, kCoordOfst>::vec((const void*)&(p[i]));
        off -= r.min();
        off /= pitch;

        // Now 'off' is a Dim-dimensional index of a finest-level cell to which the particle belong.
        // 
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
        assert(node.anchor >= 0);
# if 1   
        if (!(node.anchor < (1 << SFC::MAX_DEPTH))) {
            TAPAS_LOG_ERROR() << "Anchor, " << node.anchor
                              << ", exceeds the maximum depth." << std::endl
                              << "Particle at "
                              << ParticlePosOffset<Dim, FP, kCoordOfst>::vec((const void*)&(p[i]))
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
void SortBodies(const typename TSP::Body *b, typename TSP::Body *sorted,
                const HelperNode<TSP> *sorted_nodes,
                tapas::index_t nb) {
  for (index_t i = 0; i < nb; ++i) {
    sorted[i] = b[sorted_nodes[i].p_index];
  }
}

/**
 * @brief Returns the range of bodies from an array of T (body type) that belong to the cell specified by the given key. 
 * @tparam BT Body type. (might be replaced by Iter::value_type)
 * @tparam Iter Iterator type of the body array.
 * @tparam Functor Functor type that retrieves key from a body type value.
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


template <int DIM, class SFC, class T>
void AppendChildren(typename SFC::KeyType x, T &s) {
  using KeyType = typename SFC::KeyType;
  
  int x_depth = SFC::GetDepth(x);
  int c_depth = x_depth + 1;
  if (c_depth > SFC::MAX_DEPTH) return;
  x = SFC::IncrementDepth(x, 1);
  for (int i = 0; i < (1 << DIM); ++i) {
    KeyType child_key = ((KeyType)i << ((SFC::MAX_DEPTH - c_depth) * DIM +
                                        SFC::DEPTH_BIT_WIDTH));
    s.push_back(x | child_key);
    TAPAS_LOG_DEBUG() << "Adding child " << (x | child_key) << std::endl;
  }
}

template <class TSP>
void CompleteRegion(typename TSP::SFC::KeyType x,
                    typename TSP::SFC::KeyType y,
                    typename TSP::SFC::KeyVector &s) {
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;
  
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

// template <class TSP>
// template <class Funct>
// void Cell<TSP>::Map(Funct f, Cell<TSP> &cell) {
//   f(cell);
// }

// template <class TSP>
// template <class Funct>
// void Cell<TSP>::Map(Funct f, Cell<TSP> &c1, Cell<TSP> &c2) {
//   f(c1, c2);
// }

template <class TSP>
void Cell<TSP>::PostOrderMap(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f) {
  for (int i = 0; i < c.nsubcells(); i++) {
    auto &chld = c.subcell(i);
    PostOrderMap(chld, f);
  }
  f(c);
}

template <class TSP>
void Cell<TSP>::PreOrderMap(Cell<TSP> &c, std::function<void(Cell<TSP>&)> f) {
  f(c);
  for (int i = 0; i < c.nsubcells(); i++) {
    auto &chld = c.subcell(i);
    PreOrderMap(chld, f);
  }
}

template <class TSP>
bool Cell<TSP>::operator==(const Cell &c) const {
    return key_ == c.key_;
}

template <class TSP>
bool Cell<TSP>::operator<(const Cell &c) const {
  return key_ < c.key_;
}

template <class TSP>
bool Cell<TSP>::IsRoot() const {
  return TSP::SFC::GetDepth(key_) == 0;
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
  using SFC = typename TSP::SFC;
  typename SFC::KeyType k = TSP::SFC::Child(key_, idx);
  return *Lookup(k);
}


template <class TSP>
Cell<TSP> *Cell<TSP>::Lookup(typename TSP::SFC::KeyType k) const {
  auto i = data_->ht_.find(k);
  if (i != data_->ht_.end()) {
    return i->second;
  } else {
    return nullptr;
  }
}

template <class TSP>
Cell<TSP> &Cell<TSP>::parent() const {
  using SFC = typename TSP::SFC;
  using KeyType = typename SFC::KeyType;
  
  if (IsRoot()) {
    TAPAS_LOG_ERROR() << "Trying to access parent of the root cell." << std::endl;
    TAPAS_DIE();
  }
  KeyType parent_key = SFC::Parent(key_);
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
typename TSP::Body &Cell<TSP>::body(index_t idx) {
  return bodies_[this->bid_+idx];
}

template <class TSP>
const typename TSP::Body &Cell<TSP>::body(index_t idx) const {
  return bodies_[this->bid_+idx];
}

template <class TSP>
typename TSP::Body &Cell<TSP>::local_body(index_t idx) {
  return this->body(idx);
}

template <class TSP>
const typename TSP::Body &Cell<TSP>::local_body(index_t idx) const {
  return this->body(idx);
}

template <class TSP>
typename TSP::BodyAttr *Cell<TSP>::body_attrs() {
  return body_attrs_;
}

template <class TSP>
const typename TSP::BodyAttr *Cell<TSP>::body_attrs() const {
  return body_attrs_;
}

template <class TSP>
typename TSP::BodyAttr &Cell<TSP>::body_attr(index_t idx) {
  return body_attrs_[this->bid_+idx];
}

template <class TSP>
const typename TSP::BodyAttr &Cell<TSP>::body_attr(index_t idx) const {
  return body_attrs_[this->bid_+idx];
}

template <class TSP> // Tapas static params
class Partitioner {
 private:
  const int max_nb_;
  
 public:

  using KeyType = typename TSP::SFC::KeyType;
  Partitioner(unsigned max_nb): max_nb_(max_nb) {}
      
  Cell<TSP> *Partition(typename TSP::Body *b, index_t nb);
  Cell<TSP> *Partition(std::vector<typename TSP::Body> &b);
 private:
  void Refine(Cell<TSP> *c, const std::vector<HelperNode<TSP>> &hn,
              const typename TSP::Body *b, int cur_depth,
              KeyType cur_key) const;
}; // class Partitioner

/**
 * @brief Overloaded version of Partitioner::Partition
 */
template <class TSP>
Cell<TSP>*
Partitioner<TSP>::Partition(std::vector<typename TSP::Body> &b) {
    return Partitioner<TSP>::Partition(b.data(), b.size());
}

/**
 * @brief Partition the simulation space and build SFC-based octree
 * @tparam TSP Tapas static params
 * @param b Array of particles
 * @param nb Length of nb
 * @param r Geometry of the target space
 * @return The root cell of the constructed tree
 */
template <class TSP> // TSP : Tapas Static Params
Cell<TSP>*
Partitioner<TSP>::Partition(typename TSP::Body *b, index_t nb) {
    using SFC = typename TSP::SFC;
    using FP = typename TSP::FP;
    typedef typename TSP::Body Body;
    typedef typename TSP::BodyAttr BodyAttr;
    typedef Cell<TSP> CellType;
    const constexpr int kDim = TSP::Dim;
    const constexpr int kPosOffset = TSP::kBodyCoordOffset;
    
    Region<TSP> r;
    // calculate region
    {
      Vec<kDim, FP> local_max, local_min;
    
      for (index_t i = 0; i < nb; i++) {
        Vec<kDim, FP> pos = ParticlePosOffset<kDim, FP, kPosOffset>::vec(reinterpret_cast<const void*>(b+i));
        for (int d = 0; d < kDim; d++) {
          local_max = (i == 0) ? pos[d] : std::max(pos[d], local_max[d]);
          local_min = (i == 0) ? pos[d] : std::min(pos[d], local_min[d]);
        }
      }
    
      r.min() = local_min;
      r.max() = local_max;
    }
    
    
    Body *b_work = new Body[nb];
    std::vector<HelperNode<TSP>> hn = CreateInitialNodes<TSP>(b, nb, r);

    // Sort the helper nodes using SFC keys
    auto key_comp = [](const HelperNode<TSP> &lhs, const HelperNode<TSP> &rhs) {
        return lhs.key < rhs.key;
    };
    std::sort(hn.begin(), hn.end(), key_comp);

    // Sort particles to the same order of hn
    SortBodies<TSP>(b, b_work, hn.data(), hn.size());

    std::memcpy(b, b_work, sizeof(Body) * nb);
    //BodyAttr *attrs = new BodyAttr[nb];
    BodyAttr *attrs = (BodyAttr*)calloc(nb, sizeof(BodyAttr));

    KeyType root_key = 0;
    auto get_key = [](const HelperNode<TSP>& hn) { return hn.key; };
    auto kp = GetBodyRange<SFC, HelperNode<TSP>>(root_key, hn, get_key);
    assert(kp.first == (KeyType)0 && kp.second == (KeyType)nb); (void)kp;

    auto data = std::make_shared<SharedData<TSP>>();
    auto &ht = data->ht_;
    auto *root = new CellType(data, r, 0, nb, root_key, b, attrs);
    ht.insert(std::make_pair(root_key, root));
    Refine(root, hn, b, 0, 0);

    // Dump all (local) cells to a file
#ifdef TAPAS_DEBUG_DUMP
    {
      tapas::debug::DebugStream e("cells");
    
      for (auto&& iter : ht) {
        KeyType k = iter.first;
        Cell<TSP> *c = iter.second;
        e.out() << SFC::Simplify(k) << " "
                << "d=" << SFC::GetDepth(k) << " "
                << "leaf=" << c->IsLeaf() << " "
            //<< "owners=" << std::setw(2) << std::right << 0 << " "
                << "nb=" << std::setw(3) << (c->IsLeaf() ? (int)c->nb() : -1) << " "
                << "center=[" << c->center() << "] "
            //<< "next_key=" << SFC::Simplify(SFC::GetNext(k)) << " "
            //<< "parent=" << SFC::Simplify(SFC::Parent(k))  << " "
                << std::endl;
      }
    }
#endif
    
    return root;
}

template <class TSP>
void Partitioner<TSP>::Refine(Cell<TSP> *c,
                              const std::vector<HelperNode<TSP>> &hn,
                              const typename TSP::Body *b,
                              int cur_depth,
                              KeyType cur_key) const {
    const int Dim = TSP::Dim;
    using SFC = typename TSP::SFC;
    using KeyType = typename SFC::KeyType;
    //typedef typename TSP::FP FP;
    //typedef typename TSP::BT BT;
    
    TAPAS_LOG_INFO() << "Current depth: " << cur_depth << std::endl;
    if (c->nb() <= (size_t)max_nb_) {
        TAPAS_LOG_INFO() << "Small enough cell" << std::endl;
        return;
    }
    if (cur_depth >= SFC::MAX_DEPTH) {
        TAPAS_LOG_INFO() << "Reached maximum depth" << std::endl;
        return;
    }
    KeyType child_key = SFC::FirstChild(cur_key);
    index_t cur_offset = c->bid();
    index_t cur_len = c->nb();
    
    auto get_key = [](const HelperNode<TSP> &hn) { return hn.key; };
    
    for (int i = 0; i < (1 << Dim); ++i) {
        TAPAS_LOG_DEBUG() << "Child key: " << child_key << std::endl;
        auto kp = GetBodyRange<SFC, HelperNode<TSP>>(child_key,
                                                     hn.begin() + cur_offset,
                                                     hn.begin() + cur_offset + cur_len,
                                                     get_key);
        index_t child_bn = kp.second;
        
        TAPAS_LOG_DEBUG() << "Range: offset: " << cur_offset << ", length: "
                          << child_bn << "\n";
        auto child_r = c->region().PartitionBSP(i);
        auto *child_cell = new Cell<TSP>(
            c->data_, child_r, cur_offset, child_bn, child_key,
            c->bodies_, c->body_attrs_);
        c->ht().insert(std::make_pair(child_key, child_cell));
        TAPAS_LOG_DEBUG() << "Particles: \n";
        Refine(child_cell, hn, b, cur_depth+1, child_key);

        // Go to the next child
        child_key = SFC::GetNext(child_key);
        cur_offset = cur_offset + child_bn;
        cur_len = cur_len - child_bn;
    }
    c->is_leaf_ = false;
}

} // namespace single_node_hot

using tapas::iterator::CellIterator;
namespace sn = single_node_hot;

/**
 * @brief Constructs a ProductIterator for dual tree traversal of two trees
 */ 
template <class TSP>
ProductIterator<CellIterator<sn::Cell<TSP>>,
                CellIterator<sn::Cell<TSP>>>
Product(sn::Cell<TSP> &c1, sn::Cell<TSP> &c2) {
  TAPAS_LOG_DEBUG() << "Cell-Cell product\n";
  using CellType = sn::Cell<TSP>;
  using CI = tapas::iterator::CellIterator<CellType>;
  
  return ProductIterator<CI,CI>(CI(c1), CI(c2));
}

// New Tapas Static Params base class
template<int _DIM, class _FP, class _BODY_TYPE, size_t _BODY_COORD_OFST, class _BODY_ATTR, class _CELL_ATTR>
struct HOT {
  static const constexpr int Dim = _DIM;
  static const constexpr size_t kBodyCoordOffset = _BODY_COORD_OFST;
  using FP = _FP;
  using Body = _BODY_TYPE;
  using BodyAttr = _BODY_ATTR;
  using CellAttr = _CELL_ATTR;
  using SFC = tapas::sfc::Morton<_DIM, uint64_t>;
  using Vectormap = tapas::Vectormap_CPU<_DIM, _FP, _BODY_TYPE, _BODY_ATTR>;
  using Threading = tapas::threading::Default;

  template<class _CELL, class _BODY, class _LET>
  using Mapper = single_node_hot::CPUMapper<_CELL, _BODY, _LET>;
  template<class _TSP> using Partitioner = single_node_hot::Partitioner<_TSP>;
};

#ifdef __CUDACC__

template<int _DIM, class _FP, class _BODY_TYPE, size_t _BODY_COORD_OFST, class _BODY_ATTR, class _CELL_ATTR>
struct HOT_GPU {
  using SFC = tapas::sfc::Morton<_DIM, uint64_t>;
  using Vectormap = tapas::Vectormap_CPU<_DIM, _FP, _BODY_TYPE, _BODY_ATTR>;
  using Body = _BODY_TYPE;
  static const constexpr int kBodyCoordOffset = _BODY_COORD_OFST;
};

#endif // __CUDACC__

template<class _TSP>
struct Tapas2 {
  using TSP = _TSP;
  using Partitioner = typename TSP::template Partitioner<TSP>;
  using Region = tapas::Region<TSP>;
  using Cell = single_node_hot::Cell<TSP>;
  using BodyIterator = typename Cell::BodyIterator;
  using Body = typename TSP::Body;
  //using Mapper = typename TSP::template Mapper<Cell, Body, NONE>; // single node mapper does not require LET component
  
  /**
   * @brief Partition and build an octree of the target space.
   * @param b Array of body of BT::type.
   */
  static Cell *Partition(Body *b, index_t nb, int max_nb) {
    Partitioner part(max_nb);
    return part.Partition(b, nb);
  }
};


} // namespace tapas

#endif // TAPAS_SINGLE_NODE_HOT_
