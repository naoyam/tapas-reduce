#ifndef __TAPAS_HOT_EXACT_LET__
#define __TAPAS_HOT_EXACT_LET__

#include<vector>

#include <tapas/iterator.h>
#include <tapas/debug_util.h>
#include <tapas/geometry.h>
#include <tapas/hot/mapper.h>
#include <tapas/hot/let_common.h>

using tapas::debug::BarrierExec;

namespace tapas {
namespace hot {

template<class TSP> class Cell;
template<class TSP> class Partitioner;

/**
 * A set of static functions to construct LET (Locally Essential Tree)
 *
 * ExactLET puts no assumption on user's function but has more overhead instead.
 * It emulates all the behavior of user's function.
 */
template<class TSP>
struct ExactLET {
  // typedefs
  static const constexpr int Dim = TSP::Dim;
  using FP = typename TSP::FP;
  using CellType = Cell<TSP>;
  using KeyType = typename CellType::KeyType;
  using SFC = typename CellType::SFC;
  using Data = typename CellType::Data;
  using KeySet = typename Cell<TSP>::SFC::KeySet;
  using BodyType = typename CellType::BodyType;
  using BodyAttrType = typename CellType::BodyAttrType;
  using attr_type = typename CellType::attr_type; // alias for backward compatibility
  using CellAttrType = attr_type;
  using Vec = tapas::Vec<TSP::Dim, typename TSP::FP>;
  using Reg = Region<Dim, FP>;

  enum {
    MAP1_UP,
    MAP1_DOWN,
    MAP1_UNKNOWN
  };

  struct ProxyMapper;
  class ProxyCell;

  class ProxyBodyAttr : public BodyAttrType {
   public:
    ProxyBodyAttr(BodyAttrType &rhs) : BodyAttrType(rhs) {
    }

    template <class T>
    inline ProxyBodyAttr& operator=(const T &) {
      return *this;
    }

    template<class T>
    inline const ProxyBodyAttr& operator=(const T &) const {
      return *this;
    }
  }; // class ProxyBodyAttr

  class ProxyAttr : public CellAttrType {
   public:
    ProxyAttr(ProxyCell *cell) : CellAttrType(), cell_(cell) { }
    ProxyAttr(ProxyCell *cell, CellAttrType &rhs) : CellAttrType(rhs), cell_(cell) { }
    
    ProxyAttr &operator=(const ProxyAttr &rhs) {
      this->CellAttrType::operator=(rhs);
      cell_ = rhs.cell_;
      return *this;
    }

    template<class T>
    inline ProxyAttr& operator=(const T&) {
      cell_.MarkModified();
      return *this;
    }

    template<class T>
    inline const ProxyAttr& operator=(const T&) const {
      cell_->MarkModified();
      return *this;
    }

   private:
    ProxyCell *cell_;
  }; // class ProxyAttr

  /**
   *
   */
  class ProxyBody : public BodyType {
   public:
    ProxyBody(BodyAttrType &rhs) : BodyType(rhs) {
    }
  };

  /**
   * ProxyBodyIterator
   */
  class ProxyBodyIterator  {
   public:
    using CellType = ProxyCell;
    using value_type = ProxyBodyIterator;
    using attr_type = ProxyBodyAttr;
    //using Mapper = typename CellType::Mapper;
    using Mapper = ProxyMapper;

   private:
    ProxyCell *c_;
    index_t idx_;
    ProxyMapper mapper_;

   public:
    static const constexpr int kThreadSpawnThreshold = 100;

    ProxyBodyIterator(ProxyCell *c) : c_(c), idx_(0), mapper_() { }

    ProxyBodyIterator &operator*() {
      return *this;
    }

    constexpr bool SpawnTask() const { return false; }

    Mapper &mapper() { return mapper_; }
    const Mapper &mapper() const { return c_->mapper(); }

    inline int index() const { return idx_; }

    ProxyCell &cell() const {
      return *c_;
    }

    const ProxyBodyIterator &operator*() const {
      return *this;
    }

    bool operator==(const ProxyBodyIterator &x) const {
      return *c_ == *(x.c_) && idx_ == x.idx_;
    }

    bool operator<(const ProxyBodyIterator &x) const {
      if (*c_ == *x.c_) {
        return idx_ < x.idx_;
      } else {
        return *c_ < *x.c_;
      }
    }

    template<class T>
    bool operator==(const T&) const {
      return false;
    }

    /**
     * \fn bool ProxyBodyIterator::operator!=(const ProxyBodyIterator &x) const
     */
    bool operator!=(const ProxyBodyIterator &x) const {
      return !(*this == x);
    }

    /**
     * \fn ProxyBody &ProxyBodyIterator::operator++()
     */
    const ProxyBody &operator++() {
      return c_->body(++idx_);
    }

    /**
     * \fn ProxyBody &ProxyBodyIterator::operator++(int)
     */
    const ProxyBody &operator++(int) {
      return c_->body(idx_++);
    }

    ProxyBodyIterator operator+(int i) {
      ProxyBodyIterator ret = *this;
      ret.idx_ += i;
      TAPAS_ASSERT(ret.idx_ < size());
      return ret;
    }

    /**
     * \fn void ProxyBodyIterator::rewind(int idx)
     */
    void rewind(int idx) {
      idx_ = idx;
    }

    /**
     * \fn bool ProxyBodyIterator::AllowMutualInteraction(const ProxyBodyIterator &x) const;
     */
    bool AllowMutualInteraction(const ProxyBodyIterator &x) const {
      return *c_ == *(x.c_);
    }

    index_t size() const {
      return c_->nb();
    }

    bool IsLocal() const {
      return c_->IsLocal();
    }

    ProxyBodyIterator &operator+=(int n) {
      idx_ += n;
      TAPAS_ASSERT(idx_ < c_->RealCell()->nb());
      return *this;
    }

    // Returns a const pointer to (real) BodyType for read-only use.
    const BodyType *operator->() const {
      return reinterpret_cast<const BodyType*>(&(c_->body(idx_)));
    }

    const ProxyBodyAttr &attr() const {
      return c_->body_attr(idx_);
    }

  }; // class ProxyBodyIterator

  /**
   * ProxyCell
   */
  class ProxyCell {
    friend ProxyAttr;
    
   public:
    // Export same type definitions as tapas::hot::Cell does.
    using KeyType = tapas::hot::ExactLET<TSP>::KeyType;
    using SFC = tapas::hot::ExactLET<TSP>::SFC;

    using attr_type = ProxyAttr;
    using CellAttrType = ProxyAttr;
    using BodyAttrType = ProxyBodyAttr;
    using BodyType = ProxyBody;

    static const constexpr int Dim = TSP::Dim;
    using Threading = typename CellType::Threading;

    using RealCellType = CellType;

    using Mapper = ProxyMapper;

    // ctor
    ProxyCell(KeyType key, const Data &data)
        : key_(key), data_(data),
          marked_touched_(false), marked_split_(false), marked_body_(false), marked_modified_(false),
          is_local_(false), cell_(nullptr), bodies_(), body_attrs_(), attr_(this), parent_(nullptr), children_()
    {
      if (data.ht_.count(key_) > 0) {
        is_local_ = true;
        cell_ = data.ht_.at(key_);
        attr_ = ProxyAttr(this, cell_->attr());
      }
    }
    
    ~ProxyCell() {
      if (children_.size() > 0) {
        for (ProxyCell *ch : children_) {
          if (ch != nullptr) {
            delete ch;
          }
        }
        children_.clear();
      }
    }

    // for debug
    inline Reg region() const {
      return RealCellType::CalcRegion(key_, data_.region_);
    }

    inline ProxyCell &cell() { return *this; }
    inline const ProxyCell &cell() const { return *this; }

    inline Mapper &mapper() { return mapper_; }
    inline const Mapper &mapper() const { return mapper_; }

    static const constexpr bool Inspector = true;

    /**
     * bool ProxyCell::operator==(const ProxyCell &rhs) const
     */
    bool operator==(const ProxyCell &rhs) const {
      return key_ == rhs.key_;
    }

    bool operator<(const ProxyCell &rhs) const {
      return key_ < rhs.key_;
    }

    // Check if cells are split or not in 2-parameter Map
    template<class UserFunct, class...Args>
    static SplitType PredSplit2(KeyType trg_key, KeyType src_key, const Data &data, UserFunct f, Args...args) {
      ProxyCell trg_cell(trg_key, data);
      ProxyCell src_cell(src_key, data);

      f(trg_cell, src_cell, args...);

      if (trg_cell.marked_split_ && src_cell.marked_split_) {
        return SplitType::SplitBoth;
      } else if (trg_cell.marked_split_) {
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

    // Check direction (upward or downward) of 1-parameter Map
    template<class UserFunct, class...Args>
    static int PredDir1(KeyType key, const Data &data, UserFunct f, Args...args) {
      ProxyCell cell(key, data);

      f(cell, args...);

      // if cell is modified
      bool lv0_mod = cell.marked_modified_;
      // if any of the children is modified
      bool lv1_mod = 0;

      for (int i = 0; i < cell.nsubcells(); i++) {
        lv1_mod |= cell.subcell(i).marked_modified_;
      }

      if (lv0_mod && !lv1_mod) {
        // Upward
        return MAP1_UP;
      } else if (!lv0_mod && lv1_mod) {
        // Downward
        return MAP1_DOWN;
      }

      if (!lv0_mod && !lv1_mod) {
        // if cell and its children are all unmodified, try one more level.
        // This happends when downward traversal routein is written like this:
        // if (!cell.IsRoot()) {
        //    Cell &parent = cell.parent();
        //    // computation using parent.attr()
        //    cell.attr() = ...;
        // }

        ProxyCell &ch = cell.subcell(0);
        f(ch, args...);

        bool lv2_mod = 0;
        for (int i = 0; i < cell.nsubcells(); i++) {
          lv2_mod |= cell.subcell(i).marked_modified_;
        }
        
        if (!lv1_mod && lv2_mod) {
          // Downward
          return MAP1_DOWN;
        }
      }

      std::cerr << "Tapas [ERROR] Cannot detect directdion for 1-argument map" << std::endl;
      exit(-1);
      return MAP1_UNKNOWN; // Error
    }

    // TODO
    unsigned size() const {
      Touched();
      return 0;
    } // BasicCell::size() in cell.h  (always returns 1)

    /**
     * \fn Vec ProxyCell::center()
     */
    inline Vec center() const {
      Touched();
      return Cell<TSP>::CalcCenter(key_, data_.region_);
    }

    inline int depth() const {
      return SFC::GetDepth(key_);
    }

    /**
     * \brief Distance Function
     */
    inline FP Distance(ProxyCell &rhs, tapas::CenterClass) {
      return tapas::Distance<Dim, tapas::CenterClass, FP>::Calc(*this, rhs);
    }

    //inline FP Distance(Cell &rhs, tapas::Edge) {
    //  return tapas::Distance<tapas::Edge, FP>::Calc(*this, rhs);
    //}

    /**
     * \fn FP ProxyCell::width(FP d) const
     */
    inline FP width(FP d) const {
      Touched();
      return Cell<TSP>::CalcRegion(key_, data_.region_).width(d);
    }

    inline bool IsLeaf_real() const {
      if (is_local_) return cell_->IsLeaf();
      else           return data_.max_depth_ <= SFC::GetDepth(key_);
    }

    /**
     * \fn bool ProxyCell::IsLeaf() const
     */
    inline bool IsLeaf() const {
      Touched();
      return IsLeaf_real();
    }

    inline bool IsLocal() const {
      return is_local_;
      //return data_.ht_.count(key_) > 0;
    }

    inline bool IsRoot() const {
      return key_ == 0;
    }

    inline size_t local_nb() {
      return cell_ ? cell_->local_nb() : 0;
    }

    inline size_t nb() {
      Touched();
      if (is_local_) {
        return cell_->nb();
      } else {
        TAPAS_ASSERT(IsLeaf() && "Cell::nb() is not allowed for non-leaf cells.");
        Body();
        return 0;
      }
    }

    inline SubCellIterator<ProxyCell> subcells() {
      Split();
      return SubCellIterator<ProxyCell>(*this);
    }

    inline ProxyCell &subcell(int nch) {
      if (IsLeaf_real()) {
        std::cerr << "Tapas ERROR: Cell::subcell(int) is called for a leaf cell (in inspector)" << std::endl;
        abort();
      }
      
      TAPAS_ASSERT(nch < nsubcells());
      Split();

      if (children_.size() == 0) {
        int ns = nsubcells();
        children_.resize(ns, nullptr);
        for (int i = 0; i < ns; i++) {
          children_[i] = new ProxyCell(SFC::Child(key_, i), data_);
        }
      }
      return *children_[nch];
    }

    inline ProxyCell &parent() {
      TAPAS_ASSERT(key_ != 0);
      if (parent_ == nullptr) {
        parent_ = new ProxyCell(SFC::Parent(key_), data_);
      }

      return *parent_;
    }

    inline int nsubcells() const {
      Split();
      return IsLeaf_real() ? 0 : (1 << TSP::Dim);
    }

    /**
     * \fn ProxyCell::attr
     */
    const attr_type &attr() const {
      Touched();
      return attr_;
    }

    /**
     * \fn ProxyCell::bodies()
     */
    ProxyBodyIterator bodies() {
      Touched();
      return ProxyBodyIterator(this);
    }

    const ProxyBody &body(index_t idx) {
      Touched();
      if (is_local_) {
        TAPAS_ASSERT(IsLeaf() && "Cell::body() is not allowed for a non-leaf cell.");
        TAPAS_ASSERT((size_t)idx < cell_->nb() && "Body index out of bound. Check nb()." );

        if (bodies_.size() != cell_->nb()) {
          InitBodies();
        }
      } else {
        // never reach here because remote ProxyCell::nb() always returns 0 in LET mode.
        TAPAS_ASSERT(!"Tapas internal eror: ProxyCell::body_attr() must not be called in LET mode.");
      }

      TAPAS_ASSERT(idx < (index_t)bodies_.size());
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

    bool GetOptMutual() const { return data_.opt_mutual_; }

   protected:
    void Touched() const { marked_touched_ = true; }
    void Split() const  { marked_split_ = true; }
    void Body() const   { marked_body_ = true; }
    void MarkModified() { marked_modified_ = true; }

    void InitBodies() {
      if (cell_ != nullptr && cell_->nb() > 0) {
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
    mutable bool marked_split_;
    mutable bool marked_body_;
    bool marked_modified_;

    bool is_local_;

    CellType *cell_;
    std::vector<ProxyBody*> bodies_;
    std::vector<ProxyBodyAttr*> body_attrs_;
    attr_type attr_;
    
    ProxyCell *parent_;
    std::vector<ProxyCell*> children_;
    Mapper mapper_; // FIXME: create Mapper for every ProxyCell is not efficient.
  }; // end of class ProxyCell

  /**
   * @brief A dummy class of Mapper
   */
  struct ProxyMapper {
    const bool opt_mutual_;

    ProxyMapper() : opt_mutual_(false) { }

    // body
    template<class Funct, class...Args>
    inline void Map(Funct, ProxyBodyIterator &, Args...) {
      // empty
    }

    // body x body
    template<class Funct, class ...Args>
    inline void Map(Funct, ProxyBodyIterator, ProxyBodyIterator, Args...) {
      // empty
    }

    // body iter x body
    template<class Funct, class ...Args>
    inline void Map(Funct, ProxyBodyIterator, ProxyBody &, Args...) {
      // empty
    }

    // cell x cell
    template<class Funct, class ...Args>
    inline void Map(Funct, ProxyCell &, ProxyCell &, Args...) {
      // empty
    }

    // subcell iter X cell iter
    template <class Funct, class...Args>
    inline void Map(Funct, SubCellIterator<ProxyCell> &, CellIterator<ProxyCell> &, Args...) {
      // empty
    }

    // cell iter X subcell iter
    template <class Funct, class...Args>
    inline void Map(Funct, CellIterator<ProxyCell> &, SubCellIterator<ProxyCell> &, Args...) {
      // empty
    }

    // subcell iter X subcell iter
    template <class Funct, class...Args>
    inline void Map(Funct, SubCellIterator<ProxyCell> &, SubCellIterator<ProxyCell> &, Args...) {
      // empty
    }

    // subcell iter X subcell iter
    template <class Funct, class...Args>
    inline void Map(Funct, SubCellIterator<ProxyCell> &, Args...) {
      // empty
    }

    /**
     * @brief Map function f over product of two iterators
     */
    template <class Funct, class T1_Iter, class T2_Iter, class... Args>
    inline void MapP2(Funct /* f */, ProductIterator<T1_Iter, T2_Iter> /*prod*/, Args.../*args*/) {
      // empty
    }
    /**
     * @brief Map function f over product of two iterators
     */
    template <class Funct, class T1_Iter, class... Args>
    inline void MapP1(Funct /*f*/, ProductIterator<T1_Iter> /*prod*/, Args.../*args*/) {
      // empty
    }

  };



  // Note for UserFunct template parameter:
  // The template parameter `UserFunct` is used between all the functions in LET class
  // and seems that it should be included in the class template parameter list along with TSP.
  // However, it is actually not possible because LET class is declared as 'friend' in the Cell class
  // only with TSP parameter. It's impossible to declare a partial specialization to be friend.

  /**
   * \brief Traverse a virtual global tree and collect cells to be requested to other processes.
   * \param p Traget particle
   * \param key Source cell key
   * \param data Data
   * \param list_attr (output) Set of request keys of which attrs are to be sent
   * \param list_body (output) Set of request keys of which bodies are to be sent
   */
  template<class UserFunct, class...Args>
  static void Traverse(KeyType trg_key, KeyType src_key, Data &data,
                       KeySet &list_attr, KeySet &list_body,
                       std::mutex &list_attr_mutex, std::mutex &list_body_mutex,
                       UserFunct f, Args...args) {
    SCOREP_USER_REGION("LET-Traverse", SCOREP_USER_REGION_TYPE_FUNCTION);

    using Th = typename CellType::Threading;

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

    //tapas::debug::DebugStream("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << std::endl;

    if (is_src_local_leaf) {
      // the cell is local. everythig's fine. nothing to do.
      //tapas::debug::DebugStream("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " is_src_local_leaf" << std::endl;
      return; // SplitType::None;
    }
    
    if (is_src_remote_leaf) {
      // If the source cell is a remote leaf, we need it (with it's bodies).
      list_attr_mutex.lock();
      list_attr.insert(src_key);
      list_attr_mutex.unlock();

      list_body_mutex.lock();
      list_body.insert(src_key);
      list_body_mutex.unlock();
      //tapas::debug::DebugStream("traverse_count").out() << SFC::Simplify(trg_key) << " " << SFC::Simplify(src_key) << " is_src_remote_leaf" << std::endl;
      return; // SplitType::Body;
    }
    TAPAS_ASSERT(SFC::GetDepth(src_key) <= SFC::MAX_DEPTH);
    list_attr_mutex.lock();
    list_attr.insert(src_key);
    list_attr_mutex.unlock();

    // Approx/Split branch
    SplitType split = ExactLET<TSP>::ProxyCell::PredSplit2(trg_key, src_key, data, f, args...); // automated predicator object

    const constexpr size_t kNspawn = 3;
    bool to_spawn = SFC::GetDepth(trg_key) < kNspawn && SFC::GetDepth(src_key) < kNspawn;
    
    typename Th::TaskGroup tg;

    switch(split) {
      case SplitType::SplitBoth:
        if (to_spawn) {
          for (KeyType trg_ch : SFC::GetChildren(trg_key)) {
            if (ht.count(trg_ch) > 0) {
              for (KeyType src_ch : SFC::GetChildren(src_key)) {
                tg.createTask([&]() mutable {
                    Traverse(trg_ch, src_ch, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
                  });
              }
            }
          }
          tg.wait();
        } else {
          for (KeyType trg_ch : SFC::GetChildren(trg_key)) {
            if (ht.count(trg_ch) > 0) {
              for (KeyType src_ch : SFC::GetChildren(src_key)) {
                Traverse(trg_ch, src_ch, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
              }
            }
          }
        }
        break;
      case SplitType::SplitLeft:
        if (to_spawn) {
          for (KeyType ch : SFC::GetChildren(trg_key)) {
            if (ht.count(ch) > 0) {
              tg.createTask([&]() mutable {
                  Traverse(ch, src_key, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
                });
            }
          }
          tg.wait();
        } else {
          for (KeyType ch : SFC::GetChildren(trg_key)) {
            if (ht.count(ch) > 0) {
              Traverse(ch, src_key, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
            }
          }
        }
        break;

      case SplitType::None:
        break;

      case SplitType::SplitRight:
        if (to_spawn) {
          for (KeyType src_ch : SFC::GetChildren(src_key)) {
            tg.createTask([&]() mutable {
                Traverse(trg_key, src_ch, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
              });
          }
          tg.wait();
        } else {
          for (KeyType src_ch : SFC::GetChildren(src_key)) {
            Traverse(trg_key, src_ch, data, list_attr, list_body, list_attr_mutex, list_body_mutex, f, args...);
          }
        }
        break;

      case SplitType::Approx:
        list_attr_mutex.lock();
        list_attr.insert(src_key);
        list_attr_mutex.unlock();
        break;

      default: assert(0); // Never happens
    }
    return; // split;
  }

  static void ShowHistogram(const Data &data) {
    const int d = data.max_depth_;
    TAPAS_ASSERT(d <= SFC::MaxDepth());

#ifdef TAPAS_DEBUG
    const long ncells = data.ht_.size();
    const long nall   = (pow(8.0, d+1) - 1) / 7;
    BarrierExec([&](int,int) {
        std::cout << "Cells: " << ncells << std::endl;
        std::cout << "depth: " << d << std::endl;
        std::cout << "filling rate: " << ((double)ncells / nall) << std::endl;
      });
#endif

    std::vector<int> hist(d + 1, 0);
    for (auto p : data.ht_) {
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
  }

  /**
   * \brief Traverse hypothetical global tree and construct a cell list.
   */
  template<class UserFunct, class...Args>
  static void DoTraverse(CellType &root,
                         KeySet &req_keys_attr, KeySet &req_keys_body,
                         UserFunct f, Args...args) {
    SCOREP_USER_REGION("LET-Traverse", SCOREP_USER_REGION_TYPE_FUNCTION);
    MPI_Barrier(MPI_COMM_WORLD);
    double beg = MPI_Wtime();

    req_keys_attr.clear(); // cells of which attributes are to be transfered from remotes to local
    req_keys_body.clear(); // cells of which bodies are to be transfered from remotes to local

    std::mutex list_attr_mutex, list_body_mutex;

    // Construct request lists of necessary cells
    req_keys_attr.insert(root.key());

    Traverse(root.key(), root.key(), root.data(), req_keys_attr, req_keys_body, list_attr_mutex, list_body_mutex, f, args...);

    double end = MPI_Wtime();
    MPI_Barrier(MPI_COMM_WORLD);
    root.data().time_let_trav_main = end - beg;
  }

  /**
   * \brief Send request to remote processes
   */
  static void Request(Data &data,
                      KeySet &req_keys_attr, KeySet &req_keys_body,
                      std::vector<KeyType> &keys_attr_recv,
                      std::vector<KeyType> &keys_body_recv,
                      std::vector<int> &attr_src,
                      std::vector<int> &body_src) {
    const auto &ht = data.ht_;
    double bt_all, et_all, bt_comm, et_comm;

    MPI_Barrier(MPI_COMM_WORLD);
    bt_all = MPI_Wtime();

    // return values
    keys_attr_recv.clear(); // keys of which attributes are requested
    keys_body_recv.clear(); // keys of which attributes are requested

    attr_src.clear(); // Process IDs that requested attr_keys_recv[i]
    body_src.clear(); // Process IDs that requested attr_body_recv[i]

    // Local cells don't need to be transfered.
    // FIXME: here we calculate difference of sets {necessary cells} - {local cells} in a naive way.
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

#ifdef TAPAS_DEBUG_DUMP
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

    // vectorized req_keys_attr. A list of cells (keys) that the local process requires.
    // (send buffer)
    std::vector<KeyType> keys_attr_send(req_keys_attr.begin(), req_keys_attr.end());
    std::vector<KeyType> keys_body_send(req_keys_body.begin(), req_keys_body.end());

    TAPAS_ASSERT((int)data.proc_first_keys_.size() == data.mpi_size_);

    // Determine the destination process of each cell request
    std::vector<int> attr_dest = Partitioner<TSP>::FindOwnerProcess(data.proc_first_keys_, keys_attr_send);
    std::vector<int> body_dest = Partitioner<TSP>::FindOwnerProcess(data.proc_first_keys_, keys_body_send);

    MPI_Barrier(MPI_COMM_WORLD);
    bt_comm = MPI_Wtime();

    tapas::mpi::Alltoallv2(keys_attr_send, attr_dest, keys_attr_recv, attr_src, data.mpi_type_key_, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv2(keys_body_send, body_dest, keys_body_recv, body_src, data.mpi_type_key_, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    et_comm = MPI_Wtime();
    data.time_let_req_comm = et_comm - bt_comm;

#ifdef TAPAS_DEBUG_DUMP
    {
      assert(keys_body_recv.size() == body_src.size());
      tapas::debug::DebugStream e("body_keys_recv");
      for (size_t i = 0; i < keys_body_recv.size(); i++) {
        e.out() << SFC::Decode(keys_body_recv[i]) << " from " << body_src[i] << std::endl;
      }
    }
#endif

#ifdef TAPAS_DEBUG_DUMP
    BarrierExec([&](int rank, int) {
        std::cout << "rank " << rank << "  req_keys_attr.size() = " << req_keys_attr.size() << std::endl;
        std::cout << "rank " << rank << "  req_keys_body.size() = " << req_keys_body.size() << std::endl;
        std::cout << std::endl;
      });
#endif

    MPI_Barrier(MPI_COMM_WORLD);
    et_all = MPI_Wtime();
    data.time_let_req_all = et_all - bt_all;
  }


  /**
   * \brief Select cells and send response to the requesters.
   * \param data Data structure
   * \param [in,out] req_attr_keys Vector of SFC keys of cells of which attributes are sent in response
   * \param [in,out] attr_src      Vector of process ranks which requested req_attr_keys[i]
   * \param [in,out] req_leaf_keys Vector of SFC keys of leaf cells of which bodies are sent in response
   * \param [in,out] leaf_src      Vector of process ranks which requested req_leaf_keys[i]
   * \param [out] res_cell_attrs Vector of cell attributes which are recieved from remote ranks
   * \param [out] res_bodies     Vector of bodies which are received from remote ranks
   * \param [out] res_nb         Vector of number of bodies which res_cell_attrs[i] owns.
   *
   * \todo Parallelize operations
   */
  static void Response(Data &data,
                       std::vector<KeyType> &req_attr_keys, std::vector<int> &attr_src_ranks,
                       std::vector<KeyType> &req_leaf_keys, std::vector<int> &leaf_src_ranks,
                       std::vector<CellAttrType> &res_cell_attrs, std::vector<BodyType> &res_bodies, std::vector<index_t> &res_nb){

    SCOREP_USER_REGION("LET-Response", SCOREP_USER_REGION_TYPE_FUNCTION);
    // req_attr_keys : list of cell keys of which cell attributes are requested
    // req_leaf_keys : list of cell keys of which bodies are requested
    // attr_src_ranks      : source process ranks of req_attr_keys (which are response target ranks)
    // leaf_src_ranks      : source process ranks of req_leaf_keys (which are response target ranks)

    // Code regions:
    //   1. Pre-comm computation
    //   2. Communication (Alltoallv)
    //   3. Post-comm computation
    double bt_all=0, et_all=0;
    double bt=0, et=0;

    // ===== Pre-comm computation =====
    // Create and send responses to the src processes of requests.

    MPI_Barrier(MPI_COMM_WORLD);
    bt_all = MPI_Wtime();

    Partitioner<TSP>::SelectResponseCells(req_attr_keys, attr_src_ranks,
                                          req_leaf_keys, leaf_src_ranks,
                                          data.ht_);

    const auto &ht = data.ht_;
    int mpi_size = data.mpi_size_;

    // Prepare cell attributes to send to <attr_src_ranks> processes
    std::vector<KeyType> attr_keys_send = req_attr_keys; // copy (split senbuf and recvbuf)
    std::vector<int> attr_dest_ranks = attr_src_ranks;
    res_cell_attrs.clear();
    std::vector<CellAttrType> attr_sendbuf;
    Partitioner<TSP>::KeysToAttrs(attr_keys_send, attr_sendbuf, data.ht_);

    data.time_let_res_comp1 = et - bt;

    // ===== 2. communication =====
    // Send response keys and attributes
    MPI_Barrier(MPI_COMM_WORLD);
    bt = MPI_Wtime();

    tapas::mpi::Alltoallv2(attr_keys_send, attr_dest_ranks, req_attr_keys,  attr_src_ranks, data.mpi_type_key_, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv2(attr_sendbuf,   attr_dest_ranks, res_cell_attrs, attr_src_ranks, data.mpi_type_attr_, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    et = MPI_Wtime();
    data.time_let_res_attr_comm = et - bt;

#if 0
    // Dump request keys
    {
      tapas::debug::DebugStream e("attr_res_exa");
      for (size_t i = 0; i < attr_keys_send.size(); i++) {
        auto k = attr_keys_send[i];
        e.out() << SFC::Decode(k)
                << " (" << k << ")"
                << " to " << attr_dest_ranks[i] << std::endl;
      }
    }
#endif

    // ===== 3. Post-comm computation =====
    // Preapre all bodies to send to <leaf_src_ranks> processes
    // body_destは、いまのままalltoallvに渡すとエラーになる。
    // TODO: leaf_keys_send と body_dest と一緒にSortByKeysして（すでにされている？）、
    //       leaf_src_ranks を body_dest に書き換える必要がある（ループをまわす）

    std::vector<int> leaf_dest = leaf_src_ranks;         // copy
    std::vector<KeyType> leaf_keys_sendbuf = req_leaf_keys; // copy
    res_bodies.clear();

    // First, leaf_keys_sendbuf must be ordered by thier destination processes
    // (Since we need to send bodies later, leaf_keys_sendbuf must NOT be sorted ever again.)
    tapas::SortByKeys(leaf_dest, leaf_keys_sendbuf);

    std::vector<index_t> leaf_nb_sendbuf (leaf_keys_sendbuf.size()); // Cell <leaf_keys_sendbuf[i]> has <leaf_nb_sendbuf[i]> bodies.
    std::vector<BodyType> body_sendbuf;

    std::vector<int> leaf_sendcnt(mpi_size, 0); // used for <leaf_keys_sendbuf> and <leaf_nb_sendbuf>.
    std::vector<int> body_sendcnt(mpi_size, 0);    // used for <bodies_sendbuf>

    for (size_t i = 0; i < leaf_keys_sendbuf.size(); i++) {
      KeyType k = leaf_keys_sendbuf[i];
      CellType *c = ht.at(k);
      TAPAS_ASSERT(c->IsLeaf());
      leaf_nb_sendbuf[i] = c->nb();

      int dest = leaf_dest[i];
      leaf_sendcnt[dest]++;

      for (size_t bi = 0; bi < c->nb(); bi++) {
        body_sendbuf.push_back(c->body(bi));
        body_sendcnt[dest]++;
      }
    }

#ifdef TAPAS_DEBUG
    index_t nb_total  = std::accumulate(leaf_nb_sendbuf.begin(), leaf_nb_sendbuf.end(), 0);
    index_t nb_total2 = body_sendbuf.size();
    index_t nb_total3 = std::accumulate(body_sendcnt.begin(), body_sendcnt.end(), 0);

    TAPAS_ASSERT(nb_total  == nb_total2);
    TAPAS_ASSERT(nb_total2 == nb_total3);
#endif

    res_nb.clear();

    // This information is not necessary because source ranks of boides can be computed from
    // leaf_src_ranks_ranks and res_nb.
    std::vector<int> leaf_recvcnt; // we don't use this
    std::vector<int> body_recvcnt; // we don't use this

    MPI_Barrier(MPI_COMM_WORLD);
    bt = MPI_Wtime();

    // Send response keys and bodies
    tapas::mpi::Alltoallv(leaf_keys_sendbuf, leaf_sendcnt, req_leaf_keys, leaf_recvcnt, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(leaf_nb_sendbuf,   leaf_sendcnt, res_nb,        leaf_recvcnt, MPI_COMM_WORLD);
    tapas::mpi::Alltoallv(body_sendbuf,      body_sendcnt, res_bodies,    body_recvcnt, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    et = MPI_Wtime();
    data.time_let_res_body_comm = et - bt;

#ifdef TAPAS_DEBUG_DUMP
    tapas::debug::BarrierExec([&](int, int) {
        std::cout << "ht.size() = " << ht.size() << std::endl;
        std::cout << "req_attr_keys.size() = " << req_attr_keys.size() << std::endl;
        std::cout << "body_sendbuf.size() = " << body_sendbuf.size() << std::endl;
        std::cout << "local_bodies.size() = " << data.local_bodies_.size() << std::endl;
        std::cout << "res_bodies.size() = " << res_bodies.size() << std::endl;
      });
#endif

    // TODO: send body attributes
    // Now we assume body_attrs from remote process is all "0" data.

    data.let_bodies_.assign(std::begin(res_bodies), std::end(res_bodies));
    data.let_body_attrs_.resize(res_bodies.size());
    bzero(&data.let_body_attrs_[0], data.let_body_attrs_.size() * sizeof(data.let_body_attrs_[0]));

    TAPAS_ASSERT(data.let_bodies_.size() == res_bodies.size());

    MPI_Barrier(MPI_COMM_WORLD);
    et_all = MPI_Wtime();
    data.time_let_res_all = et_all - bt_all;
  }

  /**
   * \breif Register response cells to local LET hash table
   * \param [in,out] data Data structure (cells are registered to data->ht_lt_)
   */
  static void Register(Data *data,
                       const std::vector<KeyType> &res_cell_attr_keys,
                       const std::vector<CellAttrType> &res_cell_attrs,
                       const std::vector<KeyType> &res_leaf_keys,
                       const std::vector<index_t> &res_nb) {
    SCOREP_USER_REGION("LET-Register", SCOREP_USER_REGION_TYPE_FUNCTION);
    MPI_Barrier(MPI_COMM_WORLD);
    double beg = MPI_Wtime();

    // Register received LET cells to local ht_let_ hash table.
    for (size_t i = 0; i < res_cell_attr_keys.size(); i++) {
      KeyType k = res_cell_attr_keys[i];
      TAPAS_ASSERT(data->ht_.count(k) == 0); // Received cell must not exit in local hash.

      Cell<TSP> *c = nullptr;

      if (data->ht_gtree_.count(k) > 0) {
        c = data->ht_gtree_.at(k);
      } else {
        c = Cell<TSP>::CreateRemoteCell(k, 0, data);
      }
      c->attr() = res_cell_attrs[i];
      c->is_leaf_ = false;
      c->nb_ = 0;
      c->bid_ = 0;
      data->ht_let_[k] = c;
    }

    TAPAS_ASSERT(res_leaf_keys.size() == res_nb.size());

    // Register received leaf cells to local ht_let_ hash table.
    index_t body_offset = 0;
    for (size_t i = 0; i < res_leaf_keys.size(); i++) {
      KeyType k = res_leaf_keys[i];
      index_t nb = res_nb[i];
      Cell<TSP> *c = nullptr;

      if (data->ht_let_.count(k) > 0) {
        // If the cell is already registered to ht_let_, the cell has attributes but not body info.
        c = data->ht_let_.at(k);
      } else if (data->ht_gtree_.count(k) > 0) {
        c = data->ht_gtree_.at(k);
      } else {
        c = Cell<TSP>::CreateRemoteCell(k, 1, data);
        data->ht_let_[k] = c;
      }

      c->is_leaf_ = true;
      c->nb_ = nb;
      c->bid_ = body_offset;

      body_offset += nb;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double end = MPI_Wtime();
    data->time_let_register = end - beg;
  }

  /**
   * \brief Build Locally essential tree
   */
  template<class UserFunct, class...Args>
  static void Exchange(CellType &root, UserFunct f, Args...args) {
    if (root.data().mpi_rank_ == 0) {
      std::cout << "Using ExactLET" << std::endl;
    }
    SCOREP_USER_REGION("LET-All", SCOREP_USER_REGION_TYPE_FUNCTION);
    double beg = MPI_Wtime();

#ifdef TAPAS_DEBUG_DUMP
    ShowHistogram(root.data());
#endif

    // Traverse
    KeySet req_cell_attr_keys; // cells of which attributes are to be transfered from remotes to local
    KeySet req_leaf_keys; // cells of which bodies are to be transfered from remotes to local

    DoTraverse(root, req_cell_attr_keys, req_leaf_keys, f, args...);

    std::vector<KeyType> res_cell_attr_keys; // cell keys of which attributes are requested
    std::vector<KeyType> res_leaf_keys; // leaf cell keys of which bodies are requested

    std::vector<int> attr_src; // Process IDs that requested attr_keys_recv[i] (output from Request())
    std::vector<int> leaf_src; // Process IDs that requested attr_body_recv[i] (output from Request())

    // Request
    Request(root.data(), req_cell_attr_keys, req_leaf_keys,
            res_cell_attr_keys, res_leaf_keys, attr_src, leaf_src);

    // Response
    std::vector<CellAttrType> res_cell_attrs;
    std::vector<BodyType> res_bodies;
    std::vector<index_t> res_nb; // number of bodies responded from remote processes
    Response(root.data(), res_cell_attr_keys, attr_src, res_leaf_keys, leaf_src, res_cell_attrs, res_bodies, res_nb);

    // Register
    Register(root.data_, res_cell_attr_keys, res_cell_attrs, res_leaf_keys, res_nb);

#ifdef TAPAS_DEBUG_DUMP
    DebugDumpCells(root.data());
#endif

    double end = MPI_Wtime();
    root.data().time_let_all = end - beg;
  }

  static void DebugDumpCells(Data &data) {
    (void)data;
#ifdef TAPAS_DEBUG_DUMP
    // Debug
    // Dump all received cells to a file
    {
      tapas::debug::DebugStream e("cells_let");
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

    {
      tapas::debug::DebugStream e("M_let");

      for (auto& iter : data.ht_let_) {
        KeyType k = iter.first;
        Cell<TSP> *c = iter.second;
        if (c == nullptr) {
          e.out() << "ERROR: " << SFC::Simplify(k) << " is NULL in hash LET." << std::endl;
        } else {
          e.out() << std::setw(20) << std::right << SFC::Simplify(c->key()) << " ";
          e.out() << std::setw(3) << c->depth() << " ";
          e.out() << (c->IsLeaf() ? "L" : "_") << " ";
          e.out() << c->attr().M << std::endl;
        }
      }
    }
#endif /* TAPAS_DEBUG_DUMP */
  }

  /**
   * \brief Inspect user function for 1-map and returns MAP1_UP, MAP1_DOWN, or MAP1_UNKNOWN
   */ 
  template<class Funct, class...Args>
  static int FindMap1Direction(CellType &c, Funct f, Args...args) {
    const KeyType kRoot = 0;

    return ProxyCell::PredDir1(kRoot, c.data(), f, args...);
  }
};

} // namespace hot

} // namespace tapas

#endif // __TAPAS_HOT_EXACT_LET__
