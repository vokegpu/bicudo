#ifndef BICUDO_PHYSICS_TYPES_HPP
#define BICUDO_PHYSICS_TYPES_HPP

#include <cstdint>

namespace bicudo {
  enum physics_runtime_type {
    CPU_SIDE = 0,
    GPU_ROCM = 1
  };

  typedef uint64_t id;
  typedef bool collided;
}

#endif