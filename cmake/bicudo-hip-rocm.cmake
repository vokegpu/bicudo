add_library(
  bicudo-hip-rocm SHARED
  ${BICUDO_HIP_ROCM_SRC_FILES}
)

target_include_directories(
  bicudo-hip-rocm PRIVATE
  "./include"
  ${ROCM_INCLUDE_DIR}
)

target_compile_options(
  bicudo-hip-rocm PRIVATE
  ${BICUDO_COMPILE_OPTIONS}
)

target_compile_definitions(
  bicudo-hip-rocm PRIVATE
  BICUDO_VERSION="v${BICUDO_VERSION}"
)

set_target_properties(
  bicudo-hip-rocm PROPERTIES
  CXX_STANDARD 17
)

install(
  TARGETS bicudo-hip-rocm
  EXPORT bicudo-hip-rocm
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  EXPORT bicudo-hip-rocm
  FILE BicudoHipROCMConfig.cmake
  NAMESPACE Bicudo::
  DESTINATION lib/cmake/bicudo
)
