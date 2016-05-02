#ifndef TAPAS_ITERATOR_H_
#define TAPAS_ITERATOR_H_

#include "tapas/logging.h"
#include "tapas/cell.h"

namespace tapas {

namespace iterator {

template <class Cell>
class BodyIterator {
  Cell &c_;
  index_t idx_;
 public:
  typedef Cell CellType;
  typedef BodyIterator value_type;
  using Body = typename CellType::Body;
  using BodyAttr = typename CellType::BodyAttr;
  using attr_type = typename CellType::BodyAttr;

#ifdef TAPAS_BODY_THREAD_SPAWN_THRESHOLD
  static const constexpr int kThreadSpawnThreshold = TAPAS_BODY_THREAD_SPAWN_THRESHOLD;
#else
  static const constexpr int kThreadSpawnThreshold = 12800;
#endif
  
  explicit BodyIterator(CellType &c) : c_(c), idx_(0) {}
  BodyIterator(const BodyIterator<Cell> &rhs) : c_(rhs.c_), idx_(rhs.idx_) { }
  inline int index() const { return idx_; } // for debugging
  inline index_t size() const {
    return c_.nb();
  }
  inline const Body &operator*() const {
    TAPAS_ASSERT(idx_ < c_.nb());
    return c_.body(idx_);
  }
  inline Body& operator*() {
    TAPAS_ASSERT(idx_ < (int)c_.nb());
    return c_.body(idx_);
  }
  inline BodyIterator<CellType>& operator+=(int n) {
    idx_ += n;
    TAPAS_ASSERT((size_t)idx_ < c_.nb());
    return *this;
  }
  inline BodyIterator<CellType> operator+(int n) {
    BodyIterator<CellType> tmp(*this);
    tmp += n;
    return tmp;
  }
  inline bool IsLocal() const {
    return c_.IsLocal();
  }
  inline const typename CellType::BodyType *operator->() const {
    return &(c_.body(idx_));
  }
  void rewind(int idx) {
    idx_ = idx;
  }
  typename CellType::BodyAttrType &attr() {
    return c_.body_attr(idx_);
  }
  inline CellType &cell() {
    return c_;
  }
  inline const CellType &cell() const {
    return c_;
  }
  inline const typename CellType::BodyType &operator++() {
    return c_.body(++idx_);
  }
  inline const typename CellType::BodyType &operator++(int) {
    return c_.body(idx_++);
  }
  inline bool operator==(const BodyIterator &x) const {
    return c_ == x.c_ && idx_ == x.idx_;
  }
  
  inline bool operator<(const BodyIterator &x) const {
    if (!(c_ == x.c_)) {
      return c_ < x.c_;
    } else {
      return idx_ < x.idx_;
    }
  }

  // TODO
  inline bool SpawnTask() const {
    return c_.local_nb() >= 1; // hard-coded
  }
  
  inline bool operator!=(const BodyIterator &x) const {
    return !operator==(x);
  }
  template <class T>
  inline bool operator==(const T &) const { return false; }
  inline bool AllowMutualInteraction(const BodyIterator &x) const {
    return c_.GetOptMutual() && c_ == x.c_;
  }
};

template <class CELL>
class CellIterator {
  CELL &c_;
 public:
  int index() const { return 0; } // dummy. to be deleted soon
  CellIterator(CELL &c): c_(c) {}
  typedef CELL value_type;
  typedef CELL CellType;
  typedef typename CELL::attr_type attr_type;
  using KeyType = typename CellType::KeyType;

  static const constexpr int kThreadSpawnThreshold = 1;
  
  CELL &operator*() {
    return c_;
  }
  const CELL &operator*() const {
    return c_;
  }
  CELL &operator++() {
    return c_;
  }
  CELL &operator++(int) {
    return c_;
  }
  
  inline bool IsLocal() const {
    return c_.IsLocal();
  }

  KeyType key() const {
    return c_.key();
  }

  attr_type &attr() {
    return c_.attr();
  }
  const attr_type &attr() const {
    return c_.attr();
  }
  bool operator==(const CellIterator &x) const {
    return c_ == x.c_;
  }
  CellIterator& operator+(int n) {
    assert(n == 0); (void)n;
    return *this;
  }
  CellIterator& operator+=(int n) {
    assert(n == 0); (void)n;
    return *this;
  }
  template <class T>
  bool operator==(const T &) const { return false; }
  void rewind(int ) {}
  int size() const {
    return 1;
  }

  template<class T>
  bool AllowMutualInteraction(const T&) const {
    return false;
  }
  bool AllowMutualInteraction(const CellIterator &x) const {
    return c_.GetOptMutual() && c_ == x.c_;
  }
  
  // TODO
  inline bool SpawnTask() const {
    const constexpr int lvspawn = 5;
    return c_.depth() <= lvspawn;
  }
}; // class CellIterator

template <class Cell>
class SubCellIterator {
  Cell &c_;
  int idx_;
 public:
  using CellType = Cell;
  using value_type = CellType;
  using attr_type = typename CellType::attr_type;
  using KeyType = typename CellType::KeyType;
  using SFC = typename CellType::SFC;

  inline SubCellIterator(CellType &c)
      : c_(c)
      , idx_(0)
  {
  }
  inline SubCellIterator(const SubCellIterator& rhs)
      : c_(rhs.c_)
      , idx_(rhs.idx_)
  {}
  inline SubCellIterator& operator=(const SubCellIterator& rhs) = delete;
  
#ifdef TAPAS_SUBCELL_THREAD_SPAWN_THRESHOLD
  static const constexpr int kThreadSpawnThreshold = TAPAS_SUBCELL_THREAD_SPAWN_THRESHOLD;
#else
  static const constexpr int kThreadSpawnThreshold = 2;
#endif
  inline bool SpawnTask() const {
    return c_.local_nb() >= 1000; // FIXME: make the number configurable
  }
  
  inline int size() const {
    if (c_.IsLeaf()) {
      return 0;
    } else {
      return 1 << CellType::Dim;
    }
  }

  inline int index() const {
    return idx_;
  }

  inline Cell &cell() {
    return c_;
  }

  inline const Cell &cell() const {
    return c_;
  }

  inline value_type &operator*() {
    return c_.subcell(idx_);
  }

  inline bool IsLocal() const {
#ifdef USE_MPI
    KeyType k = key();
    return c_.data().ht_.count(k) > 0;
#else
    return true;
#endif
  }
  
  inline KeyType key() const {
    KeyType pk = c_.key();
    KeyType ck = SFC::Child(pk, idx_);
    return ck;
  }

  // prefix increment operator
  inline SubCellIterator<CellType> &operator++() {
    idx_ = std::min(size(), idx_ + 1);
    return *this;
  }
  
  inline SubCellIterator<CellType> operator++(int) {
    SubCellIterator<CellType> dup = *this;
    ++(*this);
    return dup;
  }
  inline void rewind(int idx) {
    idx_ = idx;
  }
  inline SubCellIterator<CellType>& operator+=(int ofst) {
    idx_ += ofst;
    return *this;
  }
  inline bool operator==(const SubCellIterator &x) const {
    return c_ == x.c_;
  }
  SubCellIterator operator+(int n) {
    SubCellIterator newiter(*this);
    newiter.idx_ += n;
    return newiter;
  }
  template <class T>
  bool operator==(const T &) const { return false; }
  
  template<class T>
  bool AllowMutualInteraction(const T&) const {
    return false;
  }
  bool AllowMutualInteraction(const SubCellIterator &x) const {
    return c_.GetOptMutual() && c_ == x.c_;
  }
}; // class SubCellIterator


template <class T1, class T2=void>
class ProductIterator {
 public:
  index_t idx1_;
  index_t idx2_;
  T1 t1_;
  T2 t2_;

  ProductIterator(const T1 &t1, const T2 &t2):
      idx1_(0), idx2_(0), t1_(t1), t2_(t2) {
  }
  index_t size() const {
    return t1_.size() * t2_.size();
  }
  const typename T1::value_type &first() const {
    return *t1_;
  }
  typename T1::value_type &first() {
    return *t1_;
  }
  const typename T2::value_type &second() const {
    return *t2_;
  }
  typename T2::value_type &second() {
    return *t2_;
  }
  typename T1::attr_type &attr_first() {
    return t1_.attr();
  }
  typename T2::attr_type &attr_second() {
    return t2_.attr();
  }
  
  void operator++(int) {
    if (idx2_ + 1 == t2_.size()) {
      idx1_++;
      t1_++;
      idx2_ = 0;
      t2_.rewind(idx2_);      
    } else {
      idx2_++;
      t2_++;
    }
    //return *this;
  }
}; // class ProductIterator

template <class ITER>
class ProductIterator<ITER, void> {
 public:
  index_t idx1_;
  index_t idx2_;
  ITER t1_;
  ITER t2_;
  ProductIterator(const ITER &t1, const ITER &t2):
      idx1_(0), idx2_(0), t1_(t1), t2_(t2) {
    if (t1_.AllowMutualInteraction(t2_)) {
      TAPAS_LOG_DEBUG() << "mutual interaction\n";
#if 0 // No self interaction     
      idx2_ = 1;
      t2_.rewind(idx2_);      
#endif
    }
  }
  index_t size() const {
    if (t1_.AllowMutualInteraction(t2_)) {
      return (t1_.size() + 1) * t1_.size() / 2;      
    } else {
      return t1_.size() * t2_.size();
    }
  }
  const ITER &first() const {
    return t1_;
  }
  ITER &first() {
    return t1_;
  }
  const ITER &second() const {
    return t2_;
  }
  ITER &second() {
    return t2_;
  }
  typename ITER::attr_type &attr_first() {
    return t1_.attr();
  }
  typename ITER::attr_type &attr_second() {
    return t2_.attr();
  }
  
  void operator++(int) {
    // Note:
    // This routine is not used since product iterators are parallelized for task-based runtime
    // by product_map() in map.h
    if (idx2_ + 1 == t2_.size()) {
      idx1_++;
      t1_++;
      if (t1_.AllowMutualInteraction(t2_)) {
        idx2_ = idx1_;
      } else {
        idx2_ = 0;
      }
      t2_.rewind(idx2_);      
    } else {
      idx2_++;
      t2_++;
    }
  }
}; // class ProductIterator

// template <class T1, class T2>
// ProductIterator<T1, T2> Product(T1 t1, T2 t2) {
//   return ProductIterator<T1, T2>(t1, t2);
// }


} // namespace iterator

namespace {
using namespace iterator;
}

// <subcell, subcell>
template <class CELL>
ProductIterator<SubCellIterator<CELL>>
Product(SubCellIterator<CELL> c1, SubCellIterator<CELL> c2) {
  return ProductIterator<SubCellIterator<CELL>>(c1, c2);
}

// <subcell, cell>
template <class CELL>
ProductIterator<SubCellIterator<CELL>, CellIterator<CELL>>
Product(SubCellIterator<CELL> c1, CELL &c2) { // cand 1
  return ProductIterator<SubCellIterator<CELL>,
                         CellIterator<CELL>>(c1, c2);
}

// <cell, subcell>
template <class CELL>
ProductIterator<CellIterator<CELL>, SubCellIterator<CELL>>
Product(CELL &c1, SubCellIterator<CELL> c2) { // cand 1
  return ProductIterator<CellIterator<CELL>, SubCellIterator<CELL>>(c1, c2);
}

// For any other object-object combination
template<class IterType>
ProductIterator<IterType>
Product(IterType iter1, IterType iter2) {
  return ProductIterator<IterType>(iter1, iter2);
}

#if 0 /* to be deleted. */

template <class CELL>
ProductIterator<BodyIterator<CELL>>
Product(BodyIterator<CELL> c1, BodyIterator<CELL> c2) {
  return ProductIterator<BodyIterator<CELL>>(c1, c2);
}

// cell, cell
template <class Cell>
ProductIterator<CellIterator<Cell>, CellIterator<Cell>>
Product(Cell &c1, Cell &c2) {
  typedef CellIterator<Cell> CellIterType;
  return ProductIterator<CellIterType, CellIterType>(CellIterType(c1), CellIterType(c2));
}

#endif /* to be deleted */

// template <class T1, class Cell>
// ProductIterator<T1, CellIterator<Cell>>
//                          Product(T1 t1, Cell &c) {
//     TAPAS_LOG_DEBUG() << "X-Cell product\n";
//     typedef CellIterator<Cell> CellIterType;
//     return ProductIterator<T1, CellIterType>(t1, CellIterType(c));
// }

} // namespace tapas

#endif // TAPAS_ITERATOR_H_ 
