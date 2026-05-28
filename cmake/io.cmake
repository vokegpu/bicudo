set(BICUDO_COMPILE_OPTIONS "")

if(
    CMAKE_CXX_COMPILER_ID STREQUAL "Clang"
    OR
    CMAKE_CXX_COMPILER_ID STREQUAL "GNU"
  )
  set(BICUDO_COMPILE_OPTIONS "-O3")
endif()

if(LINUX OR ANDROID)
  file(GLOB ROCM_INCLUDE_DIR "/opt/rocm/include")
  set(LIBRARY_OUTPUT_PATH "../lib/linux/")
elseif(WIN32)
  set(ROCM_HIP_DIR ${HIP_PATH})
  if (DEFINED $ENV{HIP_PATH})
    set(ROCM_HIP_DIR $ENV{HIP_PATH})
    message(STATUS "Bicudo is using HIP_PATH venv")
  endif()

  file(GLOB ROCM_INCLUDE_DIR "${ROCM_HIP_DIR}/include")
  set(LIBRARY_OUTPUT_PATH "../lib/windows/")
endif()
