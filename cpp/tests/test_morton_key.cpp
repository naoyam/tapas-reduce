#include <vector>

#include <tapas/test.h>
#include <tapas/map.h>
#include <tapas/sfc_morton.h>

SETUP_TEST;

template<class SFC>
typename SFC::KeyType GenKey(std::vector<int> args) {
  using KeyType = typename SFC::KeyType;
  
  if (args.size() == 0) {
    return (KeyType) 0;
  }

  KeyType k = 0;
  for (auto idx : args) {
    k = SFC::Child(k, idx);
  }
  return k;
}

void Test_Morton_Parent() {
  const constexpr int Dim = 3;
  using K = tapas::sfc::Morton<Dim, uint64_t>;
  using KeyType = K::KeyType;

  KeyType c = GenKey<K>({0, 1, 2});
  KeyType p = GenKey<K>({0, 1});
  ASSERT_EQ(p, K::Parent(c));
}

void Test_Morton_Next() {
  const constexpr int Dim = 3;
  using K = tapas::sfc::Morton<Dim, uint64_t>;
  using KeyType = K::KeyType;
  
  KeyType k1 = GenKey<K>({0, 1, 1});
  KeyType k2 = GenKey<K>({0, 1, 2});
  ASSERT_EQ(k2, K::GetNext(k1));

  KeyType k3 = GenKey<K>({0, 1, 7});
  KeyType k4 = GenKey<K>({0, 2, 0});
  ASSERT_EQ(k4, K::GetNext(k3));
}

void Test_Morton_IsDescendant() {
  const constexpr int Dim = 3;
  using K = tapas::sfc::Morton<Dim, uint64_t>;
  using KeyType = K::KeyType;

  // Check IsDescendant works.

  KeyType a = GenKey<K>({0, 1});
  KeyType a2 = GenKey<K>({0, 2});
  KeyType d = GenKey<K>({0, 1, 3});
  KeyType d2 = GenKey<K>({0, 1, 2, 3, 4, 5});
  
  ASSERT_TRUE(K::IsDescendant(a, d));
  ASSERT_TRUE(K::IsDescendant(a, d2));

  // If a and d are reversed
  ASSERT_FALSE(K::IsDescendant(d, a));
  ASSERT_FALSE(K::IsDescendant(d2, a));
  
  ASSERT_FALSE(K::IsDescendant(a2, d));
  ASSERT_FALSE(K::IsDescendant(a2, d2));
}

void Test_Morton_LETRequirement() {
  const constexpr int Dim = 2;
  using K = tapas::sfc::Morton<Dim, uint64_t>;
  using KeyType = K::KeyType;
  
  auto comp = [](KeyType a, KeyType b) {
    return K::RemoveDepth(a) < K::RemoveDepth(b);
  };
  
  // When given a key, we need to find the owner process
  // We have an array of first keys of processes
  std::vector<KeyType> procs = {GenKey<K>({0, 0}),  // 00-00 (2)
                                GenKey<K>({0, 3}),  // 00-11 (2)
                                GenKey<K>({3, 0})}; // 11-00 (2)

  
  // We want to find which process owns trg. (answer is 1).
  KeyType trg = GenKey<K>({2, 3}); // 10-11 (2)
  
  auto p = std::upper_bound(procs.begin(), procs.end(), trg, comp) - 1;
  //if (*p != trg) { p--; }
  ASSERT_EQ(1, p - procs.begin());

  trg = GenKey<K>({3, 0}); // 11-00
  p = std::upper_bound(procs.begin(), procs.end(), trg, comp) - 1;
  //if (*p != trg) { p--; }
  ASSERT_EQ(2, p - procs.begin());
}

void Test_Morton_GetDirOnDepth() {
  // calculate center of a cell of the key
  const constexpr int Dim = 3;
  using K = tapas::sfc::Morton<Dim, uint64_t>;
  using KeyType = K::KeyType;

  // zyx
  // 000 = 0
  // 001 = 1
  // 010 = 2
  // 011 = 3
  // 100 = 4
  // 101 = 5
  // 110 = 6
  // 111 = 7

  KeyType k = GenKey<K>({0, 0, 0});

  for (int dim = 0; dim < Dim; dim++) {
    for (int dep = 1; dep <= 3; dep++) {
      ASSERT_EQ(0, K::GetDirOnDepth(k, dim, dep));
    }
  }
  //std::cout << "--------------" << std::endl;

  k = GenKey<K>({7, 3, 2}); // 111 - 011 - 010

  //std::cout << ((K::RemoveDepth(k) >> (K::MAX_DEPTH - 1) * Dim) & 7) << std::endl;
  // Check X-dim
  ASSERT_EQ(1, K::GetDirOnDepth(k, 0, 1));
  ASSERT_EQ(1, K::GetDirOnDepth(k, 0, 2));
  ASSERT_EQ(0, K::GetDirOnDepth(k, 0, 3));

  // Check Y-dim
  ASSERT_EQ(1, K::GetDirOnDepth(k, 1, 1));
  ASSERT_EQ(1, K::GetDirOnDepth(k, 1, 2));
  ASSERT_EQ(1, K::GetDirOnDepth(k, 1, 3));

  // Check Z-dim
  ASSERT_EQ(1, K::GetDirOnDepth(k, 2, 1));
  ASSERT_EQ(0, K::GetDirOnDepth(k, 2, 2));
  ASSERT_EQ(0, K::GetDirOnDepth(k, 2, 3));
}


int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  
  Test_Morton_Parent();
  Test_Morton_Next();
  Test_Morton_IsDescendant();
  Test_Morton_LETRequirement();
  Test_Morton_GetDirOnDepth();

  TEST_REPORT_RESULT();
  
  MPI_Finalize();
  return (TEST_SUCCESS() ? 0 : 1);
}
