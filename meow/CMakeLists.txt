cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(MEOW_VERSION 0.1.0)

if(
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    OR
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
  )
  set(
    COMPILER_STATIC_INTERNAL_LIBRARIES
    -static-libgcc
    -static-libstdc++
  )
endif()

if(LINUX)
  file(GLOB ROCM_NATIVE_LIBRARY_PATH "/opt/rocm/lib/libamdhip64.so")
  file(GLOB ROCM_INCLUDE_DIR "/opt/rocm/include")
  file(GLOB BICUDO_NATIVE_LIBRARY_PATH "../lib/linux/libbicudo.a")

  include_directories(/usr/include/freetype2)

  set(
    MEOW_LINKED_LIBRARIES
    GL
    GLEW
    SDL2main
    SDL2
    freetype
    ekg
    ${ROCM_NATIVE_LIBRARY_PATH}
    ${BICUDO_NATIVE_LIBRARY_PATH}
  )
elseif(WIN32)
  set(ROCM_HIP_DIR ${HIP_PATH})
  if (DEFINED $ENV{HIP_PATH})
    set(ROCM_HIP_DIR $ENV{HIP_PATH})
    message(STATUS "Meow is using HIP_PATH venv")
  endif()

  file(GLOB ROCM_NATIVE_LIBRARY_PATH "${ROCM_HIP_DIR}/lib/amdhip64.lib")
  file(GLOB ROCM_INCLUDE_DIR "${ROCM_HIP_DIR}/include")
  file(GLOB BICUDO_NATIVE_LIBRARY_PATH "../lib/windows/libbicudo.a")

  set(
    MEOW_LINKED_LIBRARIES
    SDL2main
    SDL2
    opengl32
    glew32
    freetype
    ekg
    ${ROCM_NATIVE_LIBRARY_PATH}
    ${BICUDO_NATIVE_LIBRARY_PATH}
  )
endif()

set(EXECUTABLE_OUTPUT_PATH "../bin/")
file(GLOB_RECURSE MEOW_SRC_FILES "./src/*.cpp")

project(
  meow
  VERSION ${MEOW_VERSION}
  DESCRIPTION "Meow is a OpenGL 4-based client for implement Bicudo physics library"
)

include_directories(
  ./src
  ${ROCM_INCLUDE_DIR}
  ../include # Bicudo include dir
)

add_executable(
  meow
  ${MEOW_SRC_FILES}
)

target_link_libraries(
  meow
  ${MEOW_LINKED_LIBRARIES}
  ${COMPILER_STATIC_INTERNAL_LIBRARIES}
)