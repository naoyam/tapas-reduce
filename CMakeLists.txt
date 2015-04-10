cmake_minimum_required(VERSION 2.8)

project(Tapas)

find_package(MPI REQUIRED)

find_package(Doxygen)

# --------------------------------------------------------------------------------
# Find MassiveThreads
# MassiveThreads is available via https://code.google.com/p/massivethreads/
# TODO: split this part as a FindMassiveThreads module.

find_path(MYTH_INCLUDE_DIR
  NAMES "mtbb/task_group.h"
  PATHS $ENV{MYTH_ROOT}/include ${MYHT_ROOT}/include /usr/local/include)
set(MYTH_INCLUDE_DIRS ${MYTH_INCLUDE_DIR})
    
find_library(MYTH_LIBRARY
  NAMES "myth-native"
  PATHS $ENV{MYTH_ROOT}/lib ${MYTH_ROOT}/lib /usr/local/lib
  STATIC)
set(MYTH_LIBRARIES ${MYTH_LIBRARY})
message(STATUS ${MYTH_LIBRARY})

mark_as_advanced(MYTH_INCLUDE_DIRS MYTH_LIBRARIES)

if (MYTH_INCLUDE_DIR AND MYTH_LIBRARY)
  message(STATUS "Found MassiveThreads in ${MYTH_INCLUDE_DIR}")
  set(MYTH_FOUND 1)
endif()

# --------------------------------------------------------------------------------

message(STATUS "Compiler = ${CMAKE_CXX_COMPILER_ID}")

set(TAPAS_FLAGS_DEBUG "-DTAPAS_LOG_LEVEL=3 -DTAPAS_DEBUG=1")
set(TAPAS_FLAGS_RELEASE "-DTAPAS_LOG_LEVEL=0 -DTAPAS_DEBUG=0")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  # Debug/Release common cxxflags
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fcolor-diagnostics")
  
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 ${TAPAS_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -O0 -ggdb3 ${TAPAS_FLAGS_DEBUG}")
  
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Wno-unused-local-typedefs -fopenmp")
  
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O2 ${TAPAS_FLAGS_RELEASE} ")
  set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS} -O0 -ggdb3 ${TAPAS_FLAGS_DEBUG}")
  
  # GCC has some bugs
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.1")
    message(FATAL_ERROR "Insufficient gcc version (Tapas requires >= 4.8.1, but your version is ${CMAKE_CXX_COMPILER_VERSION})")
  endif()
  
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -fast -fopenmp")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${TAPAS_FLAGS_RELEASE}")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} ${TAPAS_FLAGS_DEBUG}")
endif()

add_subdirectory(sample)
add_subdirectory(cpp)

# --------------------------------------------------------------------------------
# Generating documentation using Doxygen when available.
# --------------------------------------------------------------------------------
if (DOXYGEN_FOUND)
  set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in)
  set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doxyfile)

  configure_file(${doxyfile_in} ${doxyfile})
  
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)
endif()

# --------------------------------------------------------------------------------
# test
# --------------------------------------------------------------------------------

enable_testing()
add_test(
  NAME exafmm
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/sample/exafmm-dev-13274dd4ac68/examples
  COMMAND python test.py
  )
