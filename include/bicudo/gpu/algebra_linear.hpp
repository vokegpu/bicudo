#ifndef BICUDO_GPU_ALGEBRA_LINEAR_HPP
#define BICUDO_GPU_ALGEBRA_LINEAR_HPP

#include "rocm.hpp"

namespace bicudo::gpu {
  // sub
  // add
  // dot
  // scalar

  constexpr uint64_t max_vertices_resource {
    12
  };

  constexpr uint64_t max_edges_resource {
    12
  };

  constexpr uint64_t vec2_unity_count {
    2
  };

  constexpr uint64_t rect_resources_size {
    (bicudo::gpu::max_vertices_resource * bicudo::gpu::vec2_unity_count)
    +
    (bicudo::gpu::max_vertices_resource * bicudo::gpu::vec2_unity_count)
  };

  struct rect_t {
  public:
    float32 resources[bicudo::gpu::rect_resources_size] {
      // meow
    };
  };

  struct collision_info_t {
  public:
    float32 depth {};
    float32 normal[2] {};
    float32 start[2] {};
    float32 end[2] {};
  };
}

#endif