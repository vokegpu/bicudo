cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(MEOW_VERSION 0.1.0)

file(GLOB ROCM_NATIVE_LIBRARY_PATH "/opt/rocm/lib/libamdhip64.so")
file(GLOB ROCM_INCLUDE_DIR "/opt/rocm/include")

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
  file(GLOB BICUDO_NATIVE_LIBRARY_PATH "../lib/linux/libbicudo.a")
  include_directories(/usr/include/freetype2)

  set(
    MEOW_LINKED_LIBRARIES
    GL
    GLEW
    ${ROCM_NATIVE_LIBRARY_PATH}
    SDL2main
    SDL2
    freetype
    ekg
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