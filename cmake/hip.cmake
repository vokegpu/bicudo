include(FetchContent)

message(STATUS "Downloading HIP...")

FetchContent_Declare(
  HIP
  GIT_REPOSITORY https://github.com/rocm/hip
  GIT_TAG ${ROCM_VERSION}
  GIT_SHALLOW TRUE
  SOURCE_DIR "hip"
)

message(STATUS "Installing HIP...")

FetchContent_MakeAvailable(HIP)
FetchContent_GetProperties(HIP)
