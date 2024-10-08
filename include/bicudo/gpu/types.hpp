#ifndef BICUDO_GPU_TYPES_HPP
#define BICUDO_GPU_TYPES_HPP

#include <cstdint>

namespace bicudo {
  enum types {
    SUCCESS = 0,
    FAILED = 1,
    INDEXED = 2, // meow-client reserved
    ARRAYS = 3, // meow-client reserved
    WRITEBACK = 4,
    WRITESTORE = 5
  };

  typedef float float32_t;
  typedef uint64_t result;
}

#endif