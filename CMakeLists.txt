cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(BICUDO_VERSIION 0.1.0)

file(GLOB ROCM_NATIVE_LIBRARY_PATH "/opt/rocm/lib/libamdhip64.so")
file(GLOB ROCM_INCLUDE_DIR "/opt/rocm/include")

if(LINUX OR ANDROID)
  set(LIBRARY_OUTPUT_PATH "../lib/linux/")
elseif(WIN32)
  set(LIBRARY_OUTPUT_PATH "../lib/windows/")
endif()

file(GLOB_RECURSE BICUDO_SRC_FILES "./src/*.cpp")

project(
  bicudo
  VERSION ${BICUDO_VERSIION}
  DESCRIPTION "SAT & Newton physics library with acceleration option to ROCm/HIP"
)

include_directories(
  ./include
  ${ROCM_INCLUDE_DIR}
)

add_library(bicudo ${BICUDO_SRC_FILES})