cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(BICUDO_VERSIION 0.1.0)

if(LINUX OR ANDROID)
  file(GLOB ROCM_INCLUDE_DIR "/opt/rocm/include")
  set(LIBRARY_OUTPUT_PATH "../lib/linux/")
elseif(WIN32)
  set(ROCM_HIP_DIR ${HIP_PATH})
  if (DEFINED $ENV{HIP_PATH})
    set(ROCM_HIP_DIR $ENV{HIP_PATH})
    message(STATUS "Meow is using HIP_PATH venv")
  endif()

  file(GLOB ROCM_INCLUDE_DIR "${ROCM_HIP_DIR}/include")
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