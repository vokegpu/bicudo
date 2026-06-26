cmake_minimum_required(
  VERSION 3.30
)

message(STATUS "Starting to prepare Bicudo cmake-building-files...")

set(BICUDO_VERSION 0.1.0)
set(ROCM_VERSION rocm-7.2.2)

include(cmake/io.cmake)
include(cmake/fetch-hip.cmake)

project(
  Bicudo
  VERSION ${BICUDO_VERSION}
  DESCRIPTION "gpu-accelerated physics library"
)

include(cmake/bicudo.cmake)

if(BICUDO_HIP_ROCM)
  include(cmake/bicudo-hip-rocm.cmake)
endif()

install(
  TARGETS ${BICUDO_EXPORT_TARGETS}
  EXPORT bicudo-targets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY ./include/bicudo
  DESTINATION include
)

install(
  EXPORT bicudo-targets
  FILE BicudoConfig.cmake
  NAMESPACE Bicudo::
  DESTINATION lib/cmake/bicudo
)

message(STATUS "Successfully generated Bicudo cmake-building-files!")
