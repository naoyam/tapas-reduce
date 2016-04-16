#ifndef TAPAS_VEC_H_
#define TAPAS_VEC_H_

#include <cstdarg>
#include <initializer_list>

#include "tapas/common.h"

namespace tapas {

template <int DIM_, class FP_>
class Vec {
  FP_ x_[DIM_];
 public:
  using FP = FP_;
  static const int Dim = DIM_;
  
  Vec() {}
  explicit Vec(const FP* v) {
    for (int i = 0; i < Dim; ++i) {
      x_[i] = v[i];
    }
  }

#if 0
  template <class... Args>  
  explicit Vec(Args... args): x_{args...} {}
#endif
  Vec(std::initializer_list<FP> list) {
    std::copy(list.begin(), list.end(), x_);
  }
  
  Vec(const Vec<Dim, FP> &v) {
    for (int i = 0; i < Dim; ++i) {
      x_[i] = v[i];
    }
  }

  Vec(const FP &x1) {
    for (int i = 0; i < Dim; ++i) {
      x_[i] = x1;
    }
  }
  Vec(const FP &x1, const FP &x2):
      x_{x1, x2} {}
  Vec(const FP &x1, const FP &x2,
      const FP &x3): x_{x1, x2, x3} {}
  Vec(const FP &x1, const FP &x2,
      const FP &x3, const FP &x4):
      x_{x1, x2, x3, x4} {}

  FP norm() const {
    FP sum = 0;
    for (int i = 0; i < Dim; i++) {
      sum += x_[i] * x_[i];
    }
    return sum;
  }

  FP &operator[](int i) {
    return x_[i];
  }
  const FP &operator[](int i) const {
    return x_[i];
  }

  Vec operator-() const {
    Vec v(*this);
    for (int i = 0; i < Dim; ++i) {
      v.x_[i] = -x_[i];
    }
    return v;
  }

  Vec operator-=(const Vec &v) {
    for (int i = 0; i < Dim; ++i) {
      x_[i] -= v[i];
    }
    return *this;
  }

  Vec operator+=(const Vec &v) {
    for (int i = 0; i < Dim; ++i) {
      x_[i] += v[i];
    }
    return *this;
  }
  
  Vec operator-(const Vec &v) const {
    Vec x(*this);    
    for (int i = 0; i < Dim; ++i) {
      x[i] -= v[i];
    }
    return x;
  }

  Vec operator/=(const Vec &v) {
    for (int i = 0; i < Dim; ++i) {
      x_[i] /= v[i];
    }
    return *this;
  }

  Vec operator*(const FP &v) const {
    Vec x(*this);
    for (int i = 0; i < Dim; ++i) {
      x[i] *= v;
    }
    return x;
  }

  Vec operator*(const Vec &v) const {
    Vec x(*this);
    for (int i = 0; i < Dim; ++i) {
      x[i] *= v[i];
    }
    return x;
  }

  Vec operator/(const FP &v) const {
    Vec x(*this);
    for (int i = 0; i < Dim; ++i) {
      x[i] /= v;
    }
    return x;
  }

  template <class T>
  Vec operator/(const T &v) const {
    Vec x(*this);
    FP c = (FP)v;
    for (int i = 0; i < Dim; ++i) {
      x[i] /= c;
    }
    return x;
  }
  
  Vec operator/(const Vec &v) const {
    Vec x(*this);
    for (int i = 0; i < Dim; ++i) {
      x[i] /= v[i];
    }
    return x;
  }

  
  Vec operator+(const FP &v) const {
    Vec x(*this);
    for (int i = 0; i < Dim; ++i) {
      x[i] += v;
    }
    return x;
  }

  Vec operator+(const Vec &v) const {
    Vec x(*this);
    for (int i = 0; i < Dim; ++i) {
      x[i] += v[i];
    }
    return x;
  }
  

  Vec operator-(const FP &v) const {
    Vec x(*this);
    for (int i = 0; i < Dim; ++i) {
      x[i] -= v;
    }
    return x;
  }

  bool operator>(const FP &v) const {
    for (int i = 0; i < Dim; ++i) {
      if (!(x_[i] > v)) return false;
    }
    return true;
  }
  
  bool operator>=(const FP &v) const {
    for (int i = 0; i < Dim; ++i) {
      if (!(x_[i] >= v)) return false;
    }
    return true;
  }
  
  bool operator<(const FP &v) const {
    for (int i = 0; i < Dim; ++i) {
      if (!(x_[i] < v)) return false;
    }
    return true;
  }

  bool operator<=(const FP &v) const {
    for (int i = 0; i < Dim; ++i) {
      if (!(x_[i] <= v)) return false;
    }
    return true;
  }

  FP reduce_sum() const {
    FP sum = x_[0];
    for (int i = 1; i < Dim; ++i) {
      sum += x_[i];
    }
    return sum;
  }
  
  std::ostream &print(std::ostream &os) const {
    std::ios::fmtflags f(os.flags());
    
    for (int i = 0; i < Dim; ++i) {
      if (i == 0) {
        os << "[";
      }

      if (x_[i] > +1000000) {
        os << std::scientific;
      } else {
        os << std::fixed << std::setprecision(6) << std::showpos;
      }
      
      os << x_[i];
      
      if (i == Dim-1) {
        os << "]";
      } else {
        os << ", ";
      }
    }
    os.flags(f);
    return os;
  }

  inline void SetMax(const Vec &rhs) {
    for (int d = 0; d < Dim; d++) {
      x_[d] = std::max(x_[d], rhs.x_[d]);
    }
  }
  
  inline void SetMin(const Vec &rhs) {
    for (int d = 0; d < Dim; d++) {
      x_[d] = std::min(x_[d], rhs.x_[d]);
    }
  }
};


template <int DIM, class FP>
std::ostream &operator<<(std::ostream &os, const Vec<DIM, FP> &v) {
  return v.print(os);
}

}

#endif /* TAPAS_VEC_H_ */
