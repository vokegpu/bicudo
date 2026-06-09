# -> how-to
# include(FetchContent)
# 
# FetchContent_Declare(
#   Bicudo
#   GIT_REPOSITORY https://github.com/vokegpu/bicudo
#   GIT_TAG feature/implement-hip-support
#   GIT_SHALLOW TRUE
#   SOURCE_DIR "bicudo"
# )
# 
# include_directories(${CMAKE_CACHEFILE_DIR}/bicudo/include/)
# 
# FetchContent_MakeAvailable(Bicudo)
# FetchContent_GetProperties(Bicudo)

# dev ignore
include_directories(../cmake-install/include)
link_directories(../cmake-install/lib)
