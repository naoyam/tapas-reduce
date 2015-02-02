add_executable(unittest unittest.cpp)

include_directories(${CMAKE_SOURCE_DIR}/cpp/include)

# Find Google's test library
# (Unittests are enabled only if the library is found)

find_library(GTEST_LIBRARY NAMES gtest)
find_library(GTEST_MAIN_LIBRARY NAMES gtest_main)

if (GTEST_LIBRARY AND GTEST_MAIN_LIBRARY)
  message(STATUS "Found gtest library at ${GTEST_LIBRARY}, ${GTEST_MAIN_LIBRARY}")
  set_target_properties(unittest PROPERTIES COMPILE_FLAGS "-DTO_SERIAL")

  target_link_libraries(unittest
    pthread
    ${GTEST_LIBRARY} ${GTEST_MAIN_LIBRARY})
endif()


