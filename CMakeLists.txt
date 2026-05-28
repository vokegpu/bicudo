cmake_minimum_required(VERSION 3.10)

message(STATUS "Starting to preare Bicudo make-files...")

include(cmake/io.cmake)

set(BICUDO_VERSION 0.1.0)

project(
  BICUDO
  VERSION ${BICUDO_VERSION}
  DESCRIPTION "gpu-accelerated physics library"
)

file(GLOB_RECURSE BICUDO_SRC_FILES "./src/*.cpp")

add_library(
  bicudo ${BICUDO_SRC_FILES}
)

target_include_directories(
  bicudo PUBLIC
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_options(
  bicudo PRIVATE
  ${BICUDO_COMPILE_OPTIONS}
)

target_compile_definitions(
  bicudo PRIVATE
  BICUDO_VERSION="v${BICUDO_VERSION}"
)

set_target_properties(
  bicudo PROPERTIES
  CXX_STANDARD 17
)

message(STATUS "Successfully generated Bicudo build-make files!")
