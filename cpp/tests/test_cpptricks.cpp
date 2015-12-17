#include <vector>

#include <tapas/test.h>
#include <tapas/common.h>

SETUP_TEST;

// Check if SFINAE technique works.
// See: http://stackoverflow.com/a/87846
template<typename T, typename R, typename ...Args>
struct has_memfunc_foo {
  template <typename U, R (U::*)(Args...)> struct SFINAE { };
  template <typename U, R (U::*)(Args...) const> struct SFINAE_c { };
  template <typename U> static char Test(SFINAE<U, &U::foo>*);
  template <typename U> static char Test(SFINAE_c<U, &U::foo>*);
  template <typename U> static int Test(...);
  static const constexpr bool value = sizeof(Test<T>(nullptr)) == sizeof(char);
};

struct A {
  int foo() { return 1; }
};

struct B {
  void foo() { }
};

struct C {
  int foo(int a) { return 2 * a; }
};

struct D {
  int foo() const { return 2; }
};

void Test_Common_MemfuncChecker() {
  int a,b,c,d;

  a = has_memfunc_foo<A, int>::value;
  b = has_memfunc_foo<B, int>::value;
  c = has_memfunc_foo<C, int>::value;
  d = has_memfunc_foo<D, int>::value;
  ASSERT_EQ(1, a);
  ASSERT_EQ(0, b);
  ASSERT_EQ(0, c);
  ASSERT_EQ(1, d);

  // check if each class has a member function "void foo()"
  a = has_memfunc_foo<A, void>::value;
  b = has_memfunc_foo<B, void>::value;
  c = has_memfunc_foo<C, void>::value;
  d = has_memfunc_foo<D, void>::value;
  ASSERT_EQ(0, a);
  ASSERT_EQ(1, b);
  ASSERT_EQ(0, c);
  ASSERT_EQ(0, d);
  
  // check if each class has a member function "int foo(int)"
  a = has_memfunc_foo<A, int, int>::value;
  b = has_memfunc_foo<B, int, int>::value;
  c = has_memfunc_foo<C, int, int>::value;
  d = has_memfunc_foo<D, int, int>::value;
  ASSERT_EQ(0, a);
  ASSERT_EQ(0, b);
  ASSERT_EQ(1, c);
  ASSERT_EQ(0, d);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  
  Test_Common_MemfuncChecker();

  TEST_REPORT_RESULT();
  
  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
