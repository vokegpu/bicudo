add_library(
  bicudo SHARED
  ${BICUDO_SRC_FILES}
)

target_include_directories(
  bicudo PRIVATE
  "./include"
  ${ROCM_INCLUDE_DIR}
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

install(
  TARGETS bicudo
  EXPORT bicudo
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
  PUBLIC_HEADER DESTINATION include
)

install(
  DIRECTORY ./include/bicudo
  DESTINATION include
)

install(
  EXPORT bicudo
  FILE BicudoConfig.cmake
  NAMESPACE Bicudo::
  DESTINATION lib/cmake/bicudo
)
