// Test of function template in class template
#include <iostream>

template<int A>
struct Foo {
  template<int B>
  int foo() {
    return A * B;
  }
};

int main() {
  Foo<3> v;
  std::cout << v.foo<4>() << std::endl;
  return 0;
}
