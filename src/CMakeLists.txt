cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")

project(Skel LANGUAGES CUDA CXX)

# CXX STANDARD
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDA REQUIRED) # find CUDA package

# include CUDA directories
include_directories(../include "${CUDA_INCLUDE_DIRS}")

if (NOT DEFINED CMAKE_CUDA_STANDARD)
  set(CMAKE_CUDA_STANDARD 11)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

# include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
# conan_basic_setup()

# create variables to compilable source codes.
set(SOURCE 
  skeleton_cuda_recon.cpp
  skeleton_cuda.cpp
  skelft_core.cpp
  skelft.cu)

add_library(Skel STATIC ${SOURCE})
target_link_libraries(Skel PUBLIC ${CUDA_LIBRARIES})
             
set_target_properties(Skel
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# flag for CUDA Compilation.                      
set_target_properties(Skel
                      PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
set_target_properties(Skel PROPERTIES CUDA_ARCHITECTURES "35;50;72")