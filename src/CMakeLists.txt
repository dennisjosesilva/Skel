cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_COMPILER "/usr/bin/g++-10")
project(Skel LANGUAGES CUDA CXX)

include_directories(../include
/usr/local/cuda/include)


# CXX STANDARD
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_library(Skel STATIC
skeleton_cuda.cpp
skelft_core.cpp
skelft.cu)

target_link_libraries(Skel PUBLIC
  cuda
  cudart
 )
             
set_target_properties(Skel
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

set_target_properties(Skel PROPERTIES CUDA_ARCHITECTURES "35;50;72")

