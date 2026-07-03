cmake_minimum_required(VERSION 3.30)
include(cmake/io.cmake)

project(
  BicudoTests
  VERSION 1.0.0
  DESCRIPTION "bicudo tests"
)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Freetype REQUIRED)
find_package(Bicudo REQUIRED)

include(cmake/install-ekg.cmake)
include(cmake/fetch-hip.cmake)

file(GLOB_RECURSE BICUDO_TESTS_SRC_FILES "./src/*.cpp")

add_executable(
  bicudo-tests
  ${BICUDO_TESTS_SRC_FILES}
)

target_include_directories(
  bicudo-tests PRIVATE
  "./src"
  ${ROCM_INCLUDE_DIR}
)

set_target_properties(
  bicudo-tests PROPERTIES
  CXX_STANDARD 17
)

set(BICUDO_LINK Bicudo::bicudo)

if(BICUDO_HIP_ROCM)
  list(APPEND BICUDO_LINK Bicudo::bicudo-hip-rocm)

  target_compile_definitions(
    bicudo-tests PRIVATE
    BICUDO_HIP_ROCM
  )
endif()

target_link_libraries(
  bicudo-tests PRIVATE
  SDL2::SDL2
  SDL2::SDL2main
  OpenGL::GL
  GLEW::GLEW
  Freetype::Freetype
  ${BICUDO_LINK}
  ${ROCM_NATIVE_LIBRARY_PATH}
  ekg
  -static-libgcc -static-libstdc++
)
