cmake_minimum_required(VERSION 3.10)

project(
  BICUDO-TESTS
  VERSION 1.0.0
  DESCRIPTION "bicudo tests"
)

find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(Freetype REQUIRED)
include(cmake/install-ekg.cmake)

file(GLOB_RECURSE BICUDO_TESTS_SRC_FILES "./src/*.cpp")

add_executable(
  bicudo-tests
  ${BICUDO_TESTS_SRC_FILES}
)

set_target_properties(
  bicudo-tests PROPERTIES
  CXX_STANDARD 17
)

target_link_libraries(
  bicudo-tests PRIVATE
  SDL2::SDL2
  SDL2::SDL2main
  OpenGL::GL
  GLEW::GLEW
  Freetype::Freetype
  ekg
  -static-libgcc -static-libstdc++
)
