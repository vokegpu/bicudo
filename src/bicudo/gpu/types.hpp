#ifndef BICUDO_GPU_TYPES_HPP
#define BICUDO_GPU_TYPES_HPP

#include <cstdint>

namespace bicudo {
  enum types {
    SUCCESS = 0,
    FAILED = 1,
    INDEXED = 2,
    ARRAYS = 3
  };

  typedef uint64_t result;
}

#endif