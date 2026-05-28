include(FetchContent)

FetchContent_Declare(
  EKG
  GIT_REPOSITORY https://github.com/vokegpu/ekg
  GIT_TAG feature/textbox
  GIT_SHALLOW TRUE
  SOURCE_DIR "ekg"
)

include_directories(${CMAKE_CACHEFILE_DIR}/ekg/include/)

FetchContent_MakeAvailable(EKG)
FetchContent_GetProperties(EKG)
