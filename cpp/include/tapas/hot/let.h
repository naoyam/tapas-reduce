#ifndef __TAPAS_HOT_LET__
#define __TAPAS_HOT_LET__

#include<vector>

#include <tapas/iterator.h>
#include <tapas/debug_util.h>

using tapas::debug::BarrierExec;

#if defined(AUTO_LET_SLOW)
#  warning "Using Auto/Slow LET"
#elif defined(MANUAL_LET)
#  warning "Using Manual LET"
#elif defined(OLD_LET_TRAVERSE)
#  warning "Using old LET traverse"
#endif

namespace tapas {

#ifdef AUTO_LET_SLOW
volatile double dummy_value = 0;
#endif

namespace hot {

template<class TSP> class Cell;
template<class TSP> class Partitioner;

/**
 * Enum values of predicate function
 */
enum class SplitType {
  Approx,       // Compute using right cell's attribute
  Body,         // Compute using right cell's bodies
  SplitLeft,    // Split Left (local) cell
  SplitRight,   // Split Right (remote) cell
  None,         // Nothing. Use when a target cell isn't local in Traverse
};

#define USING_TAPAS_TYPES(TSP)                          \
  using FP = typename TSP::FP;                          \
  using BodyType = TSP::BT::type BodyType;              \
  using BodyAttrType = TSP::BT_ATTR;                    \
  using SFC = typename TSP::SFC;                        \
  using KeyType = typename CellType::KeyType;           \
  using CellType = Cell<TSP>;                           \
  using Data = typename CellType::Data

template<class TSP, class SetType>
void TraverseLET_old(typename Cell<TSP>::BodyType &p,
                     typename Cell<TSP>::KeyType trg_key,
                     typename Cell<TSP>::KeyType src_key,
                     typename Cell<TSP>::Data &data,
                     SetType &list_attr, SetType &list_body) {
  using CellType = Cell<TSP>;
  using FP = typename TSP::FP;
  using SFC = typename Cell<TSP>::SFC;
  using KeyType = typename Cell<TSP>::KeyType;

  const constexpr double theta = 0.5;

  auto &r = data.region_;
  auto &ht = data.ht_;

  // Maximum depth of the tree.
  const int max_depth = data.max_depth_;

  bool is_src_local = ht.count(src_key) != 0;
  bool is_src_local_leaf = is_src_local && ht[src_key]->IsLeaf();
  bool is_src_remote_leaf = !is_src_local && SFC::GetDepth(src_key) >= max_depth;

  
  if (is_src_local_leaf) {
    // if the source cell is a remote leaf, we need it (the cell is not longer splittable anyway).
    //Stderr("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " is_src_local_leaf" << std::endl;
    return;
  }

  if (is_src_remote_leaf) {
    // If the source cell is a remote leaf, we need it (with it's bodies).
    list_attr.insert(src_key);
    list_body.insert(src_key);
    //Stderr("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " is_src_remote_leaf" << std::endl;
    return;
  }

  // ここまでOK
  
  // the cell attributes is necessary (because traversal has come here.)
  list_attr.insert(src_key);
  TAPAS_ASSERT(SFC::GetDepth(src_key) <= SFC::MAX_DEPTH);

  auto src_child_keys = SFC::GetChildren(src_key);

  // distance function closure, which returns distance from p
  auto distR2 = [&p](const Vec<TSP::Dim, FP> &v) -> FP {
    FP dx = p.x - v[0];
    FP dy = p.y - v[1];
    FP dz = p.z - v[2];
    return dx * dx + dy * dy + dz * dz;
  };

  for (size_t i = 0; i < src_child_keys.size(); i++) {
    KeyType ckey = src_child_keys[i];
    auto ctr = CellType::CalcCenter(ckey, r);

    FP s = CellType::CalcRegion(ckey, r).width(0); // width
    FP d = std::sqrt(distR2(ctr));
    
    // Stderr("traverse_count").out() << "particle(" << SFC::Simplify(trg_key) << ") [" << p.x << "," << p.y << "," << p.z << "]"
    //                                << " " << SFC::Simplify(ckey) << " s=" << s << " d=" << d << " "
    //                                << (s/d > theta ? "SplitRight" : "Approx")
    //                                << std::endl;

    // Stderr("comp_count_hardcoded").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(ckey) << " "
    //                                      << "s=" << s << " d=" << d << std::endl;
    if (s/d > theta) { // if the cell(ckey) is close
      TraverseLET_old<TSP, SetType>(p, trg_key, ckey, data, list_attr, list_body);
    } else {
      // If i-th children is far enough from `cell`, the rest of children
      // are also `far`. Thus we don't need to traverse them recursively
      // and just need their attributes(multipole)
      if (ht.count(ckey) == 0) {
        list_attr.insert(ckey);
      }
    }
  }
  // ------ block ends here -------
  return;
}

template<class TSP, class SetType>
void TraverseLET_old_slow(typename Cell<TSP>::KeyType trg_key, typename Cell<TSP>::KeyType src_key,
                          typename Cell<TSP>::Data &data, SetType &list_attr, SetType &list_body) {
  // 2015/10/28
  // hardcodedとmanualの速度差を説明するために、古いコードにトラバースを入れる
  // （おそらくトラバースの計算量の分が速度差であろう）

  using SFC = typename Cell<TSP>::SFC;

  const auto &ht = data.ht_;

  //Stderr("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << std::endl;

  if (ht.count(trg_key) == 0) {
    return;
  }

  auto *cell = ht.at(trg_key);
  
  if(cell->IsLeaf()) {
    if (cell->nb() == 0) {
      return;
    } else {
      assert(cell->nb() == 1);
      TraverseLET_old<TSP>(cell->body(0), trg_key, src_key, data, list_attr, list_body);
    }
  } else {
    auto child = SFC::GetChildren(trg_key);
    for (auto chk : child) {
      TraverseLET_old_slow<TSP>(chk, src_key, data, list_attr, list_body);
    }
  }
}


template<class TSP>
struct InteractionPred {
  using FP = typename TSP::FP;
  using CT = Cell<TSP>;
  using BT = typename CT::BodyType;
  using KT = typename Cell<TSP>::KeyType;
  using DT = typename Cell<TSP>::Data;
  using VT = Vec<TSP::Dim, FP>;
  using SFC = typename Cell<TSP>::SFC;

  const DT &data_;

  InteractionPred(const DT &data) : data_(data) {} 

  INLINE static FP distR2(const VT& v1, const VT& v2) {
    FP dx = v1[0] - v2[0];
    FP dy = v1[1] - v2[1];
    FP dz = v1[2] - v2[2];
    return dx * dx + dy * dy + dz * dz;
  }

  INLINE FP distR2(const BT &p, const VT &v2) {
    VT v1(p.x, p.y, p.z);
    return distR2(v1, v2);
  }
  
  template<class T1>
  INLINE FP distR2(const T1 &t, const CT &c) {
    return distR2(t, c.center());
  }

  template<class T2>
  INLINE FP distR2(const CT &c, const T2 &t) {
    return distR2(c.center(), t);
  }

  template<class T1>
  INLINE FP distR2(const T1 &v1, KT k2) {
    const auto &r = data_.region_;
    return distR2(v1, CT::CalcCenter(k2, r));
  }

  template<class T2>
  INLINE FP distR2(KT k1, const T2 &v2) {
    return distR2(CT::CalcCenter(k1, data_.region_), v2);
  }

  INLINE FP distR2(KT k1, KT k2) {
    return distR2(CT::CalcCenter(k1, data_.region_),
                  CT::CalcCenter(k2, data_.region_));
  }

  INLINE bool IsLeaf(KT k) {
    if (data_.ht_.count(k) > 0) {
      return data_.ht_.at(k)->IsLeaf();
    } else {
      return data_.max_depth_ <= SFC::GetDepth(k);
    }
  }

  INLINE size_t nb(KT) { return 1; } // nb() method for remote cell always returns '1' in LET mode
  
  INLINE SplitType operator() (KT trg_key, KT src_key) {
    const constexpr FP theta = 0.5;
    const auto &ht = data_.ht_;
    TAPAS_ASSERT(data_.ht_.count(trg_key) > 0);
    const auto &c1 = *(ht.at(trg_key));

    if (!c1.IsLeaf()) {
      return SplitType::SplitLeft;
    }

    if (c1.nb() == 0) {
      return SplitType::None;
    }

    //bool is_src_local = ht.count(src_key) != 0;
    //bool is_src_local_leaf = is_src_local && ht.at(src_key)->IsLeaf();
    //bool is_src_remote_leaf = !is_src_local && SFC::GetDepth(src_key) >= data_.max_depth_;

    if (IsLeaf(src_key)) { // c2.IsLeaf()
      return SplitType::Body;
    }

    // ここまでOK

    // else
    const auto &p1 = c1.body(0);
    real_t d = std::sqrt(distR2(p1, src_key));
    real_t s = CT::CalcRegion(src_key, data_.region_).width(0);
    
    // Stderr("traverse_count").out() << "particle(" << SFC::Simplify(trg_key) << ") [" << p1.x << "," << p1.y << "," << p1.z << "]"
    //                                << " " << SFC::Simplify(src_key) << " s=" << s << " d=" << d << " "
    //                                << (s/d > theta ? "SplitRight" : "Approx")
    //                                << std::endl;
    
    // Stderr("comp_count_manual").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " "
    //                                   << "s=" << s << " d=" << d << std::endl;
    if ((s/d) < theta) {
      return SplitType::Approx;
    } else {
      return SplitType::SplitRight;
    }
  }
};

/**
 * A set of static functions to construct LET (Locally Essential Tree)
 */
template<class TSP>
struct LET {
  // typedefs
  using FP = typename TSP::FP;
  using CellType = Cell<TSP>;
  using KeyType = typename CellType::KeyType;
  using SFC = typename CellType::SFC;
  using Data = typename CellType::Data;
  using KeySet = typename Cell<TSP>::SFC::KeySet;
  using BodyType = typename CellType::BodyType;
  using BodyAttrType = typename CellType::attr_type;
  using attr_type = typename CellType::attr_type; // alias for backward compatibility
  using CellAttrType = attr_type;
  using Vec = tapas::Vec<TSP::Dim, typename TSP::FP>;

  class ProxyBodyAttr : public BodyAttrType {
   public:
    ProxyBodyAttr(BodyAttrType &rhs) : ProxyBodyAttr(rhs) {
    }
    // staticにしてみる？
    
    template <class T>
    inline ProxyBodyAttr& operator=(const T &) {
      return *this;
    }

    template<class T>
    inline const ProxyBodyAttr& operator=(const T &) const {
      return *this;
    }

#if 0
    INLINE const ProxyBodyAttr& operator=(const BodyAttrType &v) const {
      // empty
      // In Auto-LET mechanims, assignment statement (using operator=) is replaced with this overload.
      // From a point of view of LET construction, user's functions have both of necessary and unnecessary arithmetic
      // operation. We only need the necessary operations, so we expetct the compiler does the job for us.
      // Assignment to Body or BodyAttr is overloaded by this operator=(), which is empty, and
      // optimized out by the compiler.
#if defined(AUTO_LET_SLOW)
      dummy_value = v.x;
#else
#endif
      return *this;
    }
#endif

    INLINE operator BodyAttrType& () {
      return dynamic_cast<BodyAttrType&>(*this);
    }
  };

  /**
   *
   */
  class ProxyBody : public BodyType {
   public:
    ProxyBody(BodyAttrType &rhs) : BodyType(rhs) {
    }
  };

  class ProxyCell;

  /**
   * ProxyBodyIterator
   */
  class ProxyBodyIterator  {
   public:

   private:
    ProxyCell *c_;
    index_t idx_;

   public:
    ProxyBodyIterator(ProxyCell *c) : c_(c), idx_(0) { }
  
    ProxyBodyIterator &operator*() const {
      return *this;
    }

    ProxyBodyIterator &operator+=(int n) {
      idx_ += n;
      TAPAS_ASSERT(idx_ < c_->RealCell()->nb());
    }

    // Returns a const pointer to (real) BodyType for read-only use.
    const BodyType *operator->() const {
      return reinterpret_cast<const BodyType*>(&(c_->body(idx_)));
    }
  
    const ProxyBodyAttr &attr() const {
      return c_->body_attr(idx_);
    }

    template<class Funct>
    inline static void Map(Funct f, ProxyBodyIterator &p) {
      f(p);
    }
    
    template<class Funct>
    inline static void Map(Funct f, ProxyBodyIterator &&p) {
      f(p);
    }
    
  };

  /**
   * ProxyCell
   */
  class ProxyCell {
   public:
    // Export same type definitions as tapas::hot::Cell does.
    using KeyType = tapas::hot::LET<TSP>::KeyType;
    using SFC = tapas::hot::LET<TSP>::SFC;
    using attr_type = tapas::hot::LET<TSP>::attr_type;

    using BodyAttrType = ProxyBodyAttr;
    using BodyType = ProxyBody;
    
    static const constexpr int Dim = TSP::Dim;
    using Threading = typename CellType::Threading;

    using RealCellType = CellType;

    // ctor
    ProxyCell(KeyType key, const Data &data)
        : key_(key), data_(data), marked_touched_(false), marked_split_(false), marked_body_(false),
          is_local_(false), cell_(nullptr), bodies_(), body_attrs_(), attr_()
    {
      if (data.ht_.count(key_) > 0) {
        is_local_ = true;
        cell_ = data.ht_.at(key_);
      }
    }
    
    template<class Funct>
    static void Map(Funct, ProxyCell &, ProxyCell &) {
      // empty
    }

    template<class UserFunct>
    static SplitType Pred(UserFunct f, KeyType trg_key, KeyType src_key, const Data &data) {
      ProxyCell trg_cell(trg_key, data);
      ProxyCell src_cell(src_key, data);
    
      f(trg_cell, src_cell);

      if (trg_cell.marked_split_) {
        return SplitType::SplitLeft;
      } else if (src_cell.marked_split_) {
        return SplitType::SplitRight;
      } else if (src_cell.marked_body_) {
        return SplitType::Body;
      } else if (!src_cell.marked_touched_) {
        return SplitType::None;
      } else {
        return SplitType::Approx;
      }
    }

    // TODO
    unsigned size() const {
      Touched();
      return 0;
    } // BasicCell::size() in cell.h  (always returns 1)

    Vec center() {
      Touched();
      return Cell<TSP>::CalcCenter(key_, data_.region_);
    }

    FP width(FP d) const {
      Touched();
      return Cell<TSP>::CalcRegion(key_, data_.region_).width(d);
    }

    bool IsLeaf() const {
      Touched();
      if (is_local_) return cell_->IsLeaf();
      else           return data_.max_depth_ <= SFC::GetDepth(key_);
    }

    size_t nb() {
      Touched();
      if (is_local_) {
        return cell_->nb();
      } else {
        TAPAS_ASSERT(IsLeaf() && "Cell::nb() is not allowed for non-leaf cells.");
        Body();
        return 0;
      }
    }

    SubCellIterator<ProxyCell> subcells() {
      Split();
      return SubCellIterator<ProxyCell>(*this);
    }

    ProxyCell &subcell(int) {
      return *this; 
    }

    const attr_type &attr() const {
      Touched();
      if (is_local_) {
        TAPAS_ASSERT(cell_ != nullptr);
        return cell_->attr();
      } else {
        return attr_;
      }
    }
    
    ProxyBodyIterator bodies() {
      Touched();
      return ProxyBodyIterator(this);
    }

    const ProxyBody &body(index_t idx) {
      Touched();
      if (is_local_) {
        TAPAS_ASSERT(IsLeaf());
        TAPAS_ASSERT((size_t)idx < cell_->nb());
        
        if (bodies_.size() != cell_->nb()) {
          InitBodies();
        }
      } else {
        // never reach here because remote ProxyCell::nb() always returns 0 in LET mode.
        TAPAS_ASSERT(!"Tapas internal eror: ProxyCell::body_attr() must not be called in LET mode.");
      }

      TAPAS_ASSERT(idx < bodies_.size());
      return *bodies_[idx];
    }

    ProxyBodyAttr &body_attr(index_t idx) {
      Touched();
      if (is_local_) {
        TAPAS_ASSERT(IsLeaf() && "ProxyCell::body_attr() can be called only for leaf cells");
        TAPAS_ASSERT((size_t)idx < cell_->nb());
        
        if (body_attrs_.size() != cell_->nb()) {
          InitBodies();
        }
      } else {
        // never reach here because remote ProxyCell::nb() always returns 0 in LET mode.
        TAPAS_ASSERT(!"Tapas internal eror: ProxyCell::body_attr() must not be called in LET mode.");
      }
      return *body_attrs_[idx];
    }

    KeyType key() const { return key_; }
    const Data &data() const { return data_; }

    CellType *RealCell() {
      return cell_;
    }

   protected:
    void Touched() const { marked_touched_ = true; }
    void Split()   { marked_split_ = true; }
    void Body()    { marked_body_ = true; }

    void InitBodies() {
      if (cell_ != nullptr && cell_->nb() >= 0) {
        auto nb = cell_->nb();
        if (bodies_.size() != nb) {
          bodies_.resize(nb);
          body_attrs_.resize(nb);
          for (size_t i = 0; i < nb; i++) {
            bodies_[i] = reinterpret_cast<ProxyBody*>(&cell_->body(i));
            body_attrs_[i] = reinterpret_cast<ProxyBodyAttr*>(&cell_->body_attr(i));
          }
        }
      }
    }
    
   private:
    KeyType key_;
    const Data &data_;

    mutable bool marked_touched_;
    bool marked_split_;
    bool marked_body_;
    
    bool is_local_;
    
    CellType *cell_;
    std::vector<ProxyBody*> bodies_;
    std::vector<ProxyBodyAttr*> body_attrs_;
    attr_type attr_;
  }; // end of class ProxyCell

  // Note for UserFunct template parameter:
  // The template parameter `UserFunct` is used between all the functions in LET class
  // and seems that it should be included in the class template parameter list along with TSP.
  // However, it is actually not possible because LET class is declared as 'friend' in the Cell class
  // only with TSP parameter. It's impossible to declare a partial specialization to be friend.
  
  // Supporting routine for Traverse(KeyType, KeyType, Data, KeySet, KeySet);
  template<class UserFunct>
  static void Traverse(UserFunct f, std::vector<KeyType> &trg_keys, KeyType src_key,
                       Data &data, KeySet &list_attr, KeySet &list_body) {
    // Apply Traverse for each keys in trg_keys. If interaction between trg_keys[i] and src_key is 'approximate',
    // trg_keys[i+1...] will be all 'approximate'.
    for (size_t i = 0; i < trg_keys.size(); i++) {
      Traverse(f, trg_keys[i], src_key, data, list_attr, list_body);
    }
  }

  // Supporting routine for Traverse(KeyType, KeyType, Data, KeySet, KeySet);
  template<class UserFunct>
  static void Traverse(UserFunct f, KeyType trg_key, std::vector<KeyType> src_keys,
                       Data &data, KeySet &list_attr, KeySet &list_body) {
    // Apply Traverse for each keys in src_keys.
    // If interaction between trg_key and src_keys[i] is 'approximate', trg_keys[i+1...] will be all 'approximate'.
    for (size_t i = 0; i < src_keys.size(); i++) {
      Traverse(f, trg_key, src_keys[i], data, list_attr, list_body);
    }
  }

  /**
   * \brief Traverse a virtual global tree and collect cells to be requested to other processes.
   * \param p Traget particle
   * \param key Source cell key
   * \param data Data
   * \param list_attr (output) Set of request keys of which attrs are to be sent
   * \param list_body (output) Set of request keys of which bodies are to be sent
   */
  template<class UserFunct>
  static void Traverse(UserFunct f, KeyType trg_key, KeyType src_key, Data &data,
                       KeySet &list_attr, KeySet &list_body) {
    // Traverse traverses the hypothetical global tree and constructs a list of
    // necessary cells required by the local process.
    auto &ht = data.ht_; // hash table

    // (A) check if the trg cell is local (kept in this function)
    if (ht.count(trg_key) == 0) {
      return; // SplitType::None;
    }

    // Maximum depth of the tree.
    const int max_depth = data.max_depth_;

    bool is_src_local = ht.count(src_key) != 0; // CAUTION: even if is_src_local, the children are not necessarily all local.
    bool is_src_local_leaf = is_src_local && ht[src_key]->IsLeaf();
    bool is_src_remote_leaf = !is_src_local && SFC::GetDepth(src_key) >= max_depth;

    //Stderr("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << std::endl;

    if (is_src_local_leaf) {
      // the cell is local. everythig's fine. nothing to do.
      //Stderr("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " is_src_local_leaf" << std::endl;
      return; // SplitType::None;
    }

    if (is_src_remote_leaf) {
      // If the source cell is a remote leaf, we need it (with it's bodies).
      list_attr.insert(src_key);
      list_body.insert(src_key);
      //Stderr("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " is_src_remote_leaf" << std::endl;
      return; // SplitType::Body;
    }
    TAPAS_ASSERT(SFC::GetDepth(src_key) <= SFC::MAX_DEPTH);
    list_attr.insert(src_key);

    // Approx/Split branch
#if defined(MANUAL_LET)
    SplitType split = InteractionPred<TSP>(data)(trg_key, src_key);
#elif defined(AUTO_LET_SLOW)
    SplitType split = LET<TSP>::ProxyCell::Pred(f, trg_key, src_key, data); // automated predicator object
#else
    SplitType split = LET<TSP>::ProxyCell::Pred(f, trg_key, src_key, data); // automated predicator object
#endif

    switch(split) {
      case SplitType::SplitLeft:
        for (KeyType ch : SFC::GetChildren(trg_key)) {
          if (ht.count(ch) > 0) {
            Traverse(f, ch, src_key, data, list_attr, list_body);
          }
        }
        break;

      case SplitType::None:
        break;

      case SplitType::SplitRight:
        Traverse(f, trg_key, SFC::GetChildren(src_key), data, list_attr, list_body);
        break;

      case SplitType::Approx:
        list_attr.insert(src_key); // <----- (4)
        break;

      default: assert(0); // Never happens
    }
    return; // split;
  }

  template<class UserFunct>
  static void Exchange(UserFunct f, CellType &root) {
#ifdef TAPAS_MEASURE
    double beg_all, end_all;
    double beg_trv, end_trv;
    double beg_req, end_req;
    double beg_sel, end_sel;
    double beg_res, end_res;
    double beg_reg, end_reg;
#endif
  
    KeySet req_keys_attr; // cells of which attributes are to be transfered from remotes to local
    KeySet req_keys_body; // cells of which bodies are to be transfered from remotes to local

#ifdef TAPAS_MEASURE
    beg_all = MPI_Wtime();
    beg_trv = MPI_Wtime();
#endif

    const int d = root.data().max_depth_;

#ifdef TAPAS_DEBUG
    const long ncells = root.data().ht_.size();
    const long nall   = (pow(8.0, d+1) - 1) / 7;
    BarrierExec([&](int,int) {
        std::cout << "Cells: " << ncells << std::endl;
        std::cout << "depth: " << d << std::endl;
        std::cout << "filling rate: " << ((double)ncells / nall) << std::endl;
      });
#endif

    std::vector<int> hist(d + 1, 0);
    for (auto p : root.data().ht_) {
      const auto *cell = p.second;
      if (cell->IsLeaf()) {
        hist[cell->depth()]++;
      }
    }

#ifdef TAPAS_DEBUG
    BarrierExec([&](int, int) {
        std::cout << "Depth histogram" << std::endl;
        for (int i = 0; i <= d; i++) {
          std::cout << i << " " << hist[i] << std::endl;
        }
      });
#endif
    
    // Construct request lists of necessary cells
    req_keys_attr.insert(root.key());
#ifdef OLD_LET_TRAVERSE
    (void) f;

    
#if 0 // 2015/10/28 性能評価のため一時的に粒子までトラバースするように変更
    for (size_t bi = 0; bi < root.local_nb(); bi++) {
      BodyType &b = root.local_body(bi);
      TraverseLET_old<TSP, KeySet>(b, root.key(), root.key(), root.data(), req_keys_attr, req_keys_body);
    }
#else
    TraverseLET_old_slow<TSP, KeySet>(root.key(), root.key(), root.data(), req_keys_attr, req_keys_body);
#endif
    
#else
    Traverse(f, root.key(), root.key(), root.data(), req_keys_attr, req_keys_body);
#endif

    //MPI_Finalize();
    //exit(0);

#ifdef TAPAS_MEASURE
    end_trv = MPI_Wtime();
#endif
  
#ifdef TAPAS_DEBUG
    BarrierExec([&](int rank, int) {
        std::cout << "rank " << rank << "  root.local_nb() = "   << root.local_nb() << std::endl;
        std::cout << "rank " << rank << "  req_keys_attr.size() = " << req_keys_attr.size() << std::endl;
        std::cout << "rank " << rank << "  req_keys_body.size() = " << req_keys_body.size() << std::endl;
        std::cout << std::endl;
      });
#endif
  
    const auto &ht = root.data().ht_;
    auto &data = root.data();
  
    // Local cells don't need to be transfered.
    // FIXME: here we calculate difference of sets
    //   {necessary cells} - {local cells}
    // in a naive way.
    auto orig_req_keys_attr = req_keys_attr;
    req_keys_attr.clear();
    for (auto &v : orig_req_keys_attr) {
      if (ht.count(v) == 0) {
        req_keys_attr.insert(v);
      }
    }

    auto orig_req_keys_body = req_keys_body;
    for (auto &v : orig_req_keys_body) {
      if (ht.count(v) == 0) {
        req_keys_body.insert(v);
      }
    }

#ifdef TAPAS_DEBUG
    BarrierExec([&](int rank, int) {
        std::cout << "rank " << rank << "  Local cells are filtered out" << std::endl;
        std::cout << "rank " << rank << "  req_keys_attr.size() = " << req_keys_attr.size() << std::endl;
        std::cout << "rank " << rank << "  req_keys_body.size() = " << req_keys_body.size() << std::endl;
        std::cout << std::endl;
      });

    BarrierExec([&](int rank, int) {
        if (rank == 0) {
          for (size_t i = 0; i < data.proc_first_keys_.size(); i++) {
            std::cerr << "first_key[" << i << "] = " << SFC::Decode(data.proc_first_keys_[i])
                      << std::endl;
          }
        }
      });
#endif

    // The root cell (key 0) is shared by all processes. Thus the root cell is never included in the send list.
    TAPAS_ASSERT(req_keys_attr.count(0) == 0);

    // Send cell request to each other
    // Transfer req_keys_attr using MPI_Alltoallv

    // Step 1 : Exchange requests

#ifdef TAPAS_MEASURE
    beg_req = MPI_Wtime();
#endif

    // vectorized req_keys_attr. A list of cells (keys) that the local process requires.
    // (send buffer)
    std::vector<KeyType> keys_attr_send(req_keys_attr.begin(), req_keys_attr.end());
    std::vector<KeyType> keys_body_send(req_keys_body.begin(), req_keys_body.end());
  
    // BarrierExec([&](int rank, int) {
    //     if (rank == 0) {
    //       std::cout << "data.proc_first_keys = " << std::endl;
    //       for (int i = 0; i < data.mpi_size_; i++) {
    //         std::cout << "\t" << i << " : " << SFC::Decode(data.proc_first_keys_[i]) << std::endl;
    //       }
    //       std::cout << std::endl;
    //     }
    //   });

    TAPAS_ASSERT(data.proc_first_keys_.size() == data.mpi_size_);
    
    // Determine the destination process of each cell request
    std::vector<int> attr_dest = Partitioner<TSP>::FindOwnerProcess(data.proc_first_keys_, keys_attr_send);
    std::vector<int> body_dest = Partitioner<TSP>::FindOwnerProcess(data.proc_first_keys_, keys_body_send);
    
    std::vector<KeyType> keys_attr_recv; // keys of which attributes are requested
    std::vector<KeyType> keys_body_recv; // keys of which attributes are requested
  
    std::vector<int> attr_src; // Process IDs that requested attr_keys_recv[i]
    std::vector<int> body_src; // Process IDs that requested attr_body_recv[i]
    
    tapas::mpi::Alltoallv<KeyType>(keys_attr_send, attr_dest,
                                   keys_attr_recv, attr_src, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv<KeyType>(keys_body_send, body_dest,
                                   keys_body_recv, body_src, MPI_COMM_WORLD);

#ifdef TAPAS_MEASURE
    end_req = MPI_Wtime();
#endif
  
#ifdef TAPAS_DEBUG
    {
      assert(keys_body_recv.size() == body_src.size());
      Stderr e("body_keys_recv");
      for (size_t i = 0; i < keys_body_recv.size(); i++) {
        e.out() << SFC::Decode(keys_body_recv[i]) << " from " << body_src[i] << std::endl;
      }
    }
#endif

#ifdef TAPAS_MEASURE
    beg_sel = MPI_Wtime();
#endif

    // Now create and send responses to the src processes of requests.
    // (note that sender/receiver are reversed.
  
    Partitioner<TSP>::SelectResponseCells(keys_attr_recv, attr_src,
                                          keys_body_recv, body_src,
                                          data.ht_);

#ifdef TAPAS_MEASURE
    end_sel = MPI_Wtime();
#endif
  
#ifdef TAPAS_MEASURE
    beg_res = MPI_Wtime();
#endif
                     
    // swap send/recv buffer (because response is sent to the original requester)
    keys_attr_recv.swap(keys_attr_send);
    keys_body_recv.swap(keys_body_send);
    attr_src.swap(attr_dest);
    body_src.swap(body_dest);
  
    // Prepare data to be transferred
    std::vector<CellAttrType> attr_send, attr_recv;
    Partitioner<TSP>::KeysToAttrs(keys_attr_send, attr_send, data.ht_);

    std::vector<BodyType> body_send, body_recv;
    std::vector<index_t> nb_send, nb_recv;
    Partitioner<TSP>::KeysToBodies(keys_body_send, nb_send, body_send, data.ht_);

#ifdef TAPAS_DEBUG
    for (size_t i = 0; i < attr_send.size(); i++) {
      Stderr e("send_attr");
      e.out() << "k = " << keys_attr_send[i] << ", attr = ["
              << attr_send[i].x << ", "
              << attr_send[i].y << ", "
              << attr_send[i].z << ", "
              << attr_send[i].w << "]" << std::endl;
    }
#endif
  
    // Send response keys and attributes
    tapas::mpi::Alltoallv(keys_attr_send, attr_dest, keys_attr_recv, attr_src, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(attr_send, attr_dest, attr_recv, attr_src, MPI_COMM_WORLD);
  
    // Send response keys and bodies
    tapas::mpi::Alltoallv(keys_body_send, body_dest, keys_body_recv, body_src, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(nb_send,        body_dest, nb_recv,        body_src, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(body_send,      body_dest, body_recv,      body_src, MPI_COMM_WORLD);

#ifdef TAPAS_MEASURE
    end_res = MPI_Wtime();
#endif
    // TODO: send body attributes

#ifdef TAPAS_MEASURE
    beg_reg = MPI_Wtime();
#endif
  
    data.let_bodies_ = body_recv;
  
    // Register received LET cells to local ht_let_ hash table.
    for (size_t i = 0; i < keys_attr_recv.size(); i++) {
      KeyType k = keys_attr_recv[i];
      TAPAS_ASSERT(data.ht_.count(k) == 0); // Received cell must not exit in local hash.

#if TAPAS_DEBUG
      Stderr e("recv_attr");
      e.out() << "k = " << k << ", attr = ["
              << attr_recv[i].x << ", "
              << attr_recv[i].y << ", "
              << attr_recv[i].z << ", "
              << attr_recv[i].w << "]" << std::endl;
#endif

      Cell<TSP> *c = nullptr;

      if (data.ht_gtree_.count(k) > 0) {
        c = data.ht_gtree_.at(k);
      } else {
        c = Cell<TSP>::CreateRemoteCell(k, 0, root.data_);
      }
      c->attr() = attr_recv[i];
      c->is_leaf_ = false;
      c->nb_ = 0;
      data.ht_let_[k] = c;
    }

    for (size_t i = 0; i < keys_body_recv.size(); i++) {
      KeyType k = keys_body_recv[i];
      index_t nb = nb_recv[i];
      Cell<TSP> *c = nullptr;

      if (data.ht_let_.count(k) > 0) {
        // If the cell is already registered to ht_let_, the cell has attributes but not body info.
        c = data.ht_let_.at(k);
      } else if (data.ht_gtree_.count(k) > 0) {
        c = data.ht_gtree_.at(k);
      } else {
        c = Cell<TSP>::CreateRemoteCell(k, 1, root.data_);
        data.ht_let_[k] = c;
      }

      c->is_leaf_ = true;
    
      TAPAS_ASSERT(nb == 1 || nb == 0); // restriction of Tapas
      
      c->nb_ = nb;
      if (nb > 0) {
        c->remote_bodies_.assign(body_recv.begin() + i, body_recv.begin() + i + 1);
      } else {
        c->remote_bodies_.clear();
      }
    }
  
#ifdef TAPAS_MEASURE
    end_reg = MPI_Wtime();
#endif
  
#ifdef TAPAS_DEBUG
    // Debug
    // Dump all received cells to a file
    {
      Stderr e("cells_let");
      e.out() << "ht_let.size() = " << data.ht_let_.size() << std::endl;
      for (auto& iter : data.ht_let_) {
        KeyType k = iter.first;
        Cell<TSP> *c = iter.second;
        if (c == nullptr) {
          e.out() << "ERROR: " << SFC::Simplify(k) << " is NULL in hash LET." << std::endl;
        } else {
          e.out() << SFC::Simplify(k) << " "
                  << "d=" << SFC::GetDepth(k) << " "
                  << "leaf=" << c->IsLeaf() << " "
                  << "nb=" << std::setw(3) << (c->IsLeaf() ? tapas::debug::ToStr(c->nb()) : "N/A") << " "
                  << "center=[" << c->center() << "] "
                  << "next_key=" << SFC::Simplify(SFC::GetNext(k)) << " "
                  << "parent=" << SFC::Simplify(SFC::Parent(k)) << " "
                  << std::endl;
        }
      }
    }
#endif

#ifdef TAPAS_MEASURE
    end_all = MPI_Wtime();
    BarrierExec( [=](int rank, int) {
        std::cout << "time " << rank << " ExchangeLET/All "         << (end_all - beg_all) << std::endl;
        std::cout << "time " << rank << " ExchangeLET/Traverse "    << (end_trv - beg_trv) << std::endl;
        std::cout << "time " << rank << " ExchangeLET/Request "     << (end_req - beg_req) << std::endl;
        std::cout << "time " << rank << " ExchangeLET/Select "      << (end_sel - beg_sel) << std::endl;
        std::cout << "time " << rank << " ExchangeLET/Response "    << (end_res - beg_res) << std::endl;
        std::cout << "time " << rank << " ExchangeLET/Register "    << (end_reg - beg_reg) << std::endl;
      });
#endif
  }
};

} // namespace hot

namespace iterator {

// BodyIteratorの特殊化無しで、なんとか可能にならないか？
// operator-> : returns const BodyType *
// attr() : ProxyBodyAttr()
// Map側で、Cell::BodyIteratorと呼び出すようにしないと駄目なのでは？

} // namespace iterator

// Overload tapas::Map to ignore Map of product(ProxyCell::subcells, ProxyCell)

} // namespace tapas

#endif // __TAPAS_HOT_LET__

