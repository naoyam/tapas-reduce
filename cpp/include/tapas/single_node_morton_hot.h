/**
 * @file single_node_morton_hot.h
 * @brief Implements single node Morton-order HOT (Hashed Octree) implementation
 */
#ifndef TAPAS_SINGLE_NODE_MORTON_HOT_
#define TAPAS_SINGLE_NODE_MORTON_HOT_

#include "tapas/stdcbug.h"

#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <list>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <iomanip>
#include <functional>

// for debugging
#include <fstream>

#include "tapas/cell.h"
#include "tapas/bitarith.h"
#include "tapas/logging.h"
#include "tapas/debug_util.h"
#include "tapas/iterator.h"
#include "tapas/morton_common.h"

namespace tapas {

/**
 * @brief Provides Morton-order octree partitioning for shared memory single node
 */
namespace single_node_morton_hot {

using namespace morton_common;

template <int DIM>
struct HelperNode {
    KeyType key;          //!< Morton key
    Vec<DIM, int> anchor; //!< Morton-key like vector without depth information
    index_t p_index;      //!< Index of the corresponding body
    index_t np;           //!< Number of particles in a node
};

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

template <int DIM>
void CompleteRegion(KeyType x, KeyType y, KeyVector &s);

template <class TSP>    
class Partitioner;

template <class TSP> // TapasStaticParams
class Cell: public tapas::BasicCell<TSP> { 
    friend class Partitioner<TSP>;
    friend class BodyIterator<Cell>;
 public:
  typedef unordered_map<KeyType, Cell*> HashTable;
  typedef Cell<TSP> CellType;
  typedef BodyIterator<CellType> BodyIter;
  
  typedef typename TSP::ATTR attr_type;
  typedef typename TSP::BT::type BodyType;
  typedef typename TSP::BT_ATTR BodyAttrType;
  typedef typename TSP::Threading Threading;
  //typedef typename TSP::BT_ATTR body_attr_type;
  
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
    
  static void Map(Cell<TSP> &cell,
                  std::function<void(Cell<TSP>&)> f);
  static void Map(Cell<TSP> &c1, Cell<TSP> &c2,
                  std::function<void(Cell<TSP>&, Cell<TSP>&)> f);
  static void Map(BodyIter &b1, BodyIter &b2,
                  std::function<void(BodyIter&, BodyIter&)> f);

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

  int depth() const {
    return MortonKeyGetDepth(key_);
  }
  
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
 * @brief Create an array of HelperNode from bodies
 * In the first stage of tree construction, one HelperNode is create for each body.
 * @return Array of HelperNode
 * @param p A pointer to an array of bodies
 * @param np Number of bodies
 * @param r Region object
 */
template <class TSP>
std::vector<HelperNode<TSP::Dim>> CreateInitialNodes(const typename TSP::BT::type *p,
                                                     index_t np,
                                                     const Region<TSP> &r) {
    const int Dim = TSP::Dim;
    typedef typename TSP::FP FP;
    typedef typename TSP::BT BT;
    
    std::vector<HelperNode<Dim>> nodes(np);
    FP num_cell = 1 << MAX_DEPTH; // maximum number of cells in one dimension
    Vec<Dim, FP> pitch;           // possible minimum cell width
    for (int d = 0; d < Dim; ++d) {
        pitch[d] = (r.max()[d] - r.min()[d]) / num_cell;
    }

    for (index_t i = 0; i < np; ++i) {
        // First, create 1 helper cell per particle
        HelperNode<Dim> &node = nodes[i];
        node.p_index = i;
        node.np = 1;

        // Particle pos offset is the offset of each coordinate value (x,y,z) in body structure
        Vec<Dim, FP> off = ParticlePosOffset<Dim, FP, BT::pos_offset>::vec((const void*)&(p[i]));
        off -= r.min();
        off /= pitch;

        // Now 'off' is a Dim-dimensional index of a finest-level cell to which the particle belong.
        // 
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
                              << ParticlePosOffset<Dim, FP, BT::pos_offset>::vec((const void*)&(p[i]))
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

template <class TSP>
void Cell<TSP>::Map(Cell<TSP> &cell, std::function<void(Cell<TSP>&)> f) {
  f(cell);
}

template <class TSP>
void Cell<TSP>::Map(Cell<TSP> &c1, Cell<TSP> &c2, std::function<void(Cell<TSP>&, Cell<TSP>&)> f) {
  f(c1, c2);
}

template <class TSP>
void Cell<TSP>::Map(BodyIterator<Cell<TSP>> &b1, BodyIterator<Cell<TSP>> &b2,
                    std::function<void(BodyIterator<Cell<TSP>>&, BodyIterator<Cell<TSP>>&)> f) {
  f(b1, b2);
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
    void Refine(Cell<TSP> *c, const std::vector<HelperNode<TSP::Dim>> &hn,
                const typename TSP::BT::type *b, int cur_depth,
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
 * @brief Partition the simulation space and build Morton-key based octree
 * @tparam TSP Tapas static params
 * @param b Array of particles
 * @param nb Length of nb
 * @param r Geometry of the target space
 * @return The root cell of the constructed tree
 */
template <class TSP> // TSP : Tapas Static Params
Cell<TSP>*
Partitioner<TSP>::Partition(typename TSP::BT::type *b,
                            index_t nb,
                            const Region<TSP> &r) {
    const int Dim = TSP::Dim;
    typedef typename TSP::FP FP;
    typedef typename TSP::BT BT;
    typedef typename TSP::BT_ATTR BodyAttrType;
    typedef typename BT::type BodyType;
    typedef Cell<TSP> CellType;
    
    BodyType *b_work = new BodyType[nb];
    std::vector<HelperNode<Dim>> hn = CreateInitialNodes<TSP>(b, nb, r);

    // Sort the helper nodes using morton keys
    auto key_comp = [](const HelperNode<Dim> &lhs, const HelperNode<Dim> &rhs) {
        return lhs.key < rhs.key;
    };
    std::sort(hn.begin(), hn.end(), key_comp);

    // Sort particles to the same order of hn
    SortBodies<Dim, BT>(b, b_work, hn.data(), hn.size());

    std::memcpy(b, b_work, sizeof(BodyType) * nb);
    //BodyAttrType *attrs = new BodyAttrType[nb];
    BodyAttrType *attrs = (BodyAttrType*)calloc(nb, sizeof(BodyAttrType));

    KeyType root_key = 0;
    KeyPair kp = GetBodyRange<Dim>(root_key, hn,
                                   [](const HelperNode<Dim>& hn) { return hn.key; });
    assert(kp.first == 0 && kp.second == nb);

    auto *ht = new typename CellType::HashTable();
    auto *root = new CellType(r, 0, nb, root_key, ht, b, attrs);
    ht->insert(std::make_pair(root_key, root));
    Refine(root, hn, b, 0, 0);

    // Dump all (local) cells to a file
    {
      std::vector<KeyType> recv_keys;
      for (auto &i : hn) {
        recv_keys.push_back(i.key);
      }
      
      Stderr e("cells");
    
      for (auto&& iter : (*ht)) {
        KeyType k = iter.first;
        Cell<TSP> *c = iter.second;
        if (c->key() != 0) {
          e.out() << SimplifyKey(k) << " "
                  << "d=" << MortonKeyGetDepth(k) << " "
                  << "leaf=" << c->IsLeaf() << " "
                  << "owners=" << std::setw(2) << std::right << 0 << " "
                  << "nb=" << std::setw(3) << c->nb() << " "
                  << "center=[" << c->center() << "] "
                  << "next_key=" << SimplifyKey(CalcMortonKeyNext<Dim>(k)) << " "
                  << "parent=" << SimplifyKey(MortonKeyParent<Dim>(k)) 
                  << std::endl;
          // Print bodies which belong to Cell c
          if (c->IsLeaf()) {
            index_t body_beg, body_end;
            FindRangeByKey<TSP>(recv_keys, k, body_beg, body_end);
            for (int i = body_beg; i < body_end; i++) {
              e.out() << "\t\t\t| "
                      << SimplifyKey(recv_keys[i]) << ": "
                      << b[i].X
                      << std::endl;
            }
          }
        }
      }
    }
    
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

        // Go to the next child
        child_key = CalcMortonKeyNext<Dim>(child_key);
        cur_offset = cur_offset + child_bn;
        cur_len = cur_len - child_bn;
    }
    c->is_leaf_ = false;
}

} // namespace single_node_morton_hot

template <class TSP, class T2>
ProductIterator<CellIterator<single_node_morton_hot::Cell<TSP>>, T2>
Product(single_node_morton_hot::Cell<TSP> &c, T2 t2) {
    TAPAS_LOG_DEBUG() << "Cell-X product\n";
    typedef single_node_morton_hot::Cell<TSP> CellType;
    typedef CellIterator<CellType> CellIterType;
    return ProductIterator<CellIterType, T2>(CellIterType(c), t2);
}

template <class T1, class TSP>
ProductIterator<T1, CellIterator<single_node_morton_hot::Cell<TSP>>>
                         Product(T1 t1, single_node_morton_hot::Cell<TSP> &c) {
    TAPAS_LOG_DEBUG() << "X-Cell product\n";
    typedef single_node_morton_hot::Cell<TSP> CellType;
    typedef CellIterator<CellType> CellIterType;
    return ProductIterator<T1, CellIterType>(t1, CellIterType(c));
}

/**
 * @brief Constructs a ProductIterator for dual tree traversal of two trees
 */ 
template <class TSP>
ProductIterator<CellIterator<single_node_morton_hot::Cell<TSP>>,
                CellIterator<single_node_morton_hot::Cell<TSP>>>
                         Product(single_node_morton_hot::Cell<TSP> &c1,
                                 single_node_morton_hot::Cell<TSP> &c2) {
    TAPAS_LOG_DEBUG() << "Cell-Cell product\n";
    typedef single_node_morton_hot::Cell<TSP> CellType;
    typedef CellIterator<CellType> CellIterType;
    return ProductIterator<CellIterType, CellIterType>(
        CellIterType(c1), CellIterType(c2));
}

/** 
 * @brief A dummy class to achieve template specialization.
 */
struct SingleNodeMortonHOT {
};

/** 
 * @brief Advance decleration of a dummy class to achieve template specialization.
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR,
          class PartitionAlgorithm,
          class Threading>
class Tapas;

/**
 * @brief Specialization of Tapas for HOT (single node Morton HOT) algorithm
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR, class Threading>
class Tapas<DIM, FP, BT, BT_ATTR, CELL_ATTR, SingleNodeMortonHOT, Threading> {
  typedef TapasStaticParams<DIM, FP, BT, BT_ATTR, CELL_ATTR, Threading> TSP; // Tapas static params
 public:
  typedef tapas::Region<TSP> Region;  
  typedef single_node_morton_hot::Cell<TSP> Cell;
  //typedef tapas::BodyIterator<DIM, BT, BT_ATTR, Cell> BodyIterator;
  typedef tapas::BodyIterator<Cell> BodyIterator;  

  /**
   * @brief Partition and build an octree of the target space.
   * @param b Array of body of BT::type.
   */
  static Cell *Partition(typename BT::type *b,
                         index_t nb, const Region &r,
                         int max_nb) {
    single_node_morton_hot::Partitioner<TSP> part(max_nb);
    return part.Partition(b, nb, r);
  }
};

/**
 * @brief Specialization of Tapas for HOT (single node Morton HOT) algorithm
 * With default threading policy 'Serial'
 */
template <int DIM, class FP, class BT,
          class BT_ATTR, class CELL_ATTR>
class Tapas<DIM, FP, BT, BT_ATTR, CELL_ATTR, SingleNodeMortonHOT, tapas::threading::Serial> {
  typedef TapasStaticParams<DIM, FP, BT, BT_ATTR, CELL_ATTR, tapas::threading::Serial> TSP; // Tapas static params
 public:
  typedef tapas::Region<TSP> Region;  
  typedef single_node_morton_hot::Cell<TSP> Cell;
  //typedef tapas::BodyIterator<DIM, BT, BT_ATTR, Cell> BodyIterator;
  typedef tapas::BodyIterator<Cell> BodyIterator;  
  
  /**
   * @brief Partition and build an octree of the target space.
   * @param b Array of body of BT::type.
   */
  static Cell *Partition(typename BT::type *b,
                         index_t nb, const Region &r,
                         int max_nb) {
    single_node_morton_hot::Partitioner<TSP> part(max_nb);
    return part.Partition(b, nb, r);
  }
};

} // namespace tapas

#endif // TAPAS_SINGLE_NODE_MORTON_HOT_
