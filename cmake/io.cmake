set(BICUDO_COMPILE_OPTIONS "")
set(BICUDO_EXPORT_TARGETS "")

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

function(
  exclude_files_by_regex
  src_files
  regex
)
  foreach(PATH ${${src_files}})
    if(${PATH} MATCHES ${regex})
      message(STATUS "Removed: '${PATH}'")
      list(REMOVE_ITEM ${src_files} ${PATH})
    else()
      message(STATUS "Keep: '${PATH}'")
    endif()
  endforeach()

  return(
    PROPAGATE
    ${src_files}
  )
endfunction()

file(
  GLOB_RECURSE BICUDO_HIP_ROCM_SRC_FILES
  "./src/pipeline/rocm.cpp"
  "./src/pipeline/rocm/*.cpp"
)

file(
  GLOB_RECURSE BICUDO_SRC_FILES
  "./src/*.cpp"
)

## #
## Excludes the GPUs driver implementation due shareable pipeline. 
## #
exclude_files_by_regex(BICUDO_SRC_FILES "bicudo/pipeline/rocm/|rocm.cpp")
