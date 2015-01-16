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

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ggdb3 -Wall -O2")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops -fcolor-diagnostics")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -ggdb3 -Wall -O2 -Wno-unused-local-typedefs -fopenmp")
  if(CMAKE_CXX_COMPILER_VERSION VERSION_LESS "4.8.1")
    message(FATAL_ERROR "Insufficient gcc version (Tapas requires >= 4.8.1, but your version is ${CMAKE_CXX_COMPILER_VERSION})")
  endif()
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -fast -fopenmp")
endif()

add_subdirectory(sample)

# --------------------------------------------------------------------------------
# Generating documentation using Doxygen when available.
if (DOXYGEN_FOUND)
  set(doxyfile_in ${CMAKE_CURRENT_SOURCE_DIR}/doxyfile.in)
  set(doxyfile ${CMAKE_CURRENT_BINARY_DIR}/doxyfile)

  configure_file(${doxyfile_in} ${doxyfile})
  
  add_custom_target(doc
    COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/doxyfile
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    VERBATIM)
endif()