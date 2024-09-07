cmake_minimum_required(VERSION 3.9)

set(CMAKE_CXX_STANDARD 17)
set(BICUDO_VERSIION 0.1.0)

if (LINUX)
  file(GLOB BICUDO_NATIVE_LIBRARY_PATH "./lib/linux/libbicudo.a")

  set(LIBRARY_OUTPUT_PATH "../lib/linux/")
  set(EXECUTABLE_OUTPUT_PATH "../bin/linux/")
  set(BICUDO_LINKED_LIBRARIES GL GLEW)
  set(BICUDO_CLIENT_LINKED_LIBRARIES SDL2main SDL2 freetype ekg ${BICUDO_NATIVE_LIBRARY_PATH})
endif()

file(GLOB_RECURSE BICUDO_SRC_FILES "./src/bicudo/*.cpp")
file(GLOB_RECURSE BICUDO_CLIENT_SRC_FILES "./src/client/*.cpp")

project(
  bicudo
  VERSION ${BICUDO_VERSIION}
  DESCRIPTION "physics engine"
)

include_directories(./src)
add_library(bicudo ${BICUDO_SRC_FILES})

message("--- ${BICUDO_SRC_FILES}")

project(
  bicudoclient
  VERSION ${BICUDO_VERSIION}
  DESCRIPTION "bicudo client"
)

include_directories("./src/")

add_executable(bicudoclient ${BICUDO_CLIENT_SRC_FILES})
target_link_libraries(
  bicudoclient
  ${BICUDO_CLIENT_LINKED_LIBRARIES}
  -static-libgcc -static-libstdc++
)