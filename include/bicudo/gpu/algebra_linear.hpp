#ifndef BICUDO_GPU_ALGEBRA_LINEAR_HPP
#define BICUDO_GPU_ALGEBRA_LINEAR_HPP

#include "rocm.hpp"

namespace bicudo::gpu {
  struct vec2_t {
  public:
    float32 x {};
    float32 y {};
  };

  // sub
  // add
  // dot
  // scalar

  struct rect_t {
  public:
    bicudo::gpu::vec2_t vertices[12] {};
    bicudo::gpu::vec2_t edges[12] {};
  };

  struct support_info_t {
  public:
    float32 distance {};
    bicudo::gpu::vec2_t point {};
    bool32 has_support_point {};
  };
}

#endif