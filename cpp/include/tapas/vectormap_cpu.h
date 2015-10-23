/* vectormap_cpu.h -*- Coding: us-ascii-unix; -*- */

#ifndef TAPAS_VECTORMAP_CPU_H_
#define TAPAS_VECTORMAP_CPU_H_

/** @file vectormap_cpu.h @brief A mock direct part working similar
    way to the GPU implementation. */

/* NOTES: (0) This class selects an implementation of mapping on
   bodies.  It is not parametrized on the "Cell" type, because it is
   not available yet at the use of this class.  (1) It assumes some
   operations defined in the underlying datatypes.  The datatype of
   body attribute should define initializer by a constant (value=0.0)
   and assignment by "+=".  (2) It iterates over the bodies by
   "BodyIterator" directly, not via "ProductIterator".  (3) The code
   is fixed on the current definition of "AllowMutualInteraction()".
   The code should be fixed as it changes. */

namespace tapas {

template<int _DIM, class _FP, class _BT, class _BT_ATTR>
struct Vectormap_CPU {

  template <typename T>
  using um_allocator = std::allocator<T>;

  static void vectormap_setup(int cta, int nstreams) {}
  static void vectormap_release() {}

  static void vectormap_start() {}
  static void vectormap_finish() {}

#if 0
  template <class Funct, class Cell, class...Args>
  static void vector_map1(Funct f, BodyIterator<Cell> iter,
                          Args... args) {
    int sz = iter.size();
    for (int i = 0; i < sz; i++) {
      f(*(iter + i), args...);
    }
  }
#endif

  template <class Funct, class Cell, class...Args>
  static void vectormap_map_loop2(Funct f, Cell &c0, Cell &c1,
                                  Args... args) {
    typedef typename Cell::BT::type BV;
    typedef typename Cell::BT_ATTR BA;

    assert(c0.IsLeaf() && c1.IsLeaf());
    /* (Cast to drop const, below). */
    BV* v0 = (BV*)&(c0.body(0));
    BV* v1 = (BV*)&(c1.body(0));
    BA* a0 = (BA*)&(c0.body_attr(0));
    size_t n0 = c0.nb();
    size_t n1 = c1.nb();
    assert(n0 != 0 && n1 != 0);

    for (size_t i = 0; i < n0; i++) {
      BA attr = BA(0.0);
      for (size_t j = 0; j < n1; j++) {
        if (!(c0 == c1 && i == j)) {
          f((v0 + i), (v1 + j), attr, args...);
        }
      }
      *(a0 + i) += attr;
    }
  }

  template <class Funct, class Cell, class...Args>
  static void vector_map2(Funct f, ProductIterator<BodyIterator<Cell>> prod,
                          Args... args) {
    //typedef BodyIterator<Cell> Iter;
    const Cell &c0 = prod.first().cell();
    const Cell &c1 = prod.second().cell();
    /*bool equalcell = b0.AllowMutualInteraction(b1);*/
    if (c0 == c1) {
      vectormap_map_loop2(f, c0, c1, args...);
    } else {
      vectormap_map_loop2(f, c0, c1, args...);
      vectormap_map_loop2(f, c1, c0, args...);
    }
  }

};

}

#endif /*TAPAS_VECTORMAP_CPU_H_*/
