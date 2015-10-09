include_directories(${CMAKE_SOURCE_DIR}/cpp/include)

foreach(trg test_set test_morton_key test_allgatherv test_alltoallv)
  add_executable(${trg} ${trg}.cpp)

  set_property(TARGET ${trg} APPEND PROPERTY COMPILE_FLAGS       "${MPI_CXX_COMPILE_FLAGS}")
  set_property(TARGET ${trg} APPEND PROPERTY COMPILE_FLAGS       " -DTO_SERIAL -g -gdwarf-2")
  set_property(TARGET ${trg} APPEND PROPERTY INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")
  set_property(TARGET ${trg} APPEND PROPERTY LINK_FLAGS          "${MPI_CXX_LINK_FLAGS}")
  
  target_link_libraries(${trg} ${MPI_CXX_LIBRARIES})
endforeach()



