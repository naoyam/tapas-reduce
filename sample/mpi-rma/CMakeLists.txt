message(STATUS "MPI_CXX_LINK_FLAGS = ${MPI_CXX_LINK_FLAGS}")
message(STATUS "MPI_CXX_LIBRARIES = ${MPI_CXX_LIBRARIES}")

add_executable(rma-test rma-test.cxx)

set_property(TARGET rma-test PROPERTY INCLUDE_DIRECTORIES ${MPI_CXX_INCLUDE_PATH})
target_link_libraries(rma-test ${MPI_LIBRARIES})

if(MPI_COMPILE_FLAGS)
  set_target_properties(rma-test PROPERTIES COMPILE_FLAGS "${MPI_COMPILE_FLAGS}")
endif()

if(MPI_LINK_FLAGS)
  set_target_properties(rma-test PROPERTIES LINK_FLAGS "${MPI_LINK_FLAGS}")
endif() 

