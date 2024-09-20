#ifndef BICUDO_GPU_ALGEBRA_LINEAR_HPP
#define BICUDO_GPU_ALGEBRA_LINEAR_HPP

#include "rocm.hpp"

namespace bicudo::gpu {
  // sub
  // add
  // dot
  // scalar

  struct stride_t {
  public:
    uint64_t offset {};
    uint64_t size {};
  };

  constexpr bicudo::gpu::stride_t it_best_distance {0, 1};
  constexpr bicudo::gpu::stride_t it_best_edge {1, 2};
  constexpr bicudo::gpu::stride_t it_has_support_point {2, 1};
  constexpr bicudo::gpu::stride_t it_depth {3, 1};
  constexpr bicudo::gpu::stride_t it_normal {4, 2};
  constexpr bicudo::gpu::stride_t it_start {6, 2};
  constexpr bicudo::gpu::stride_t it_end {8, 2};
  constexpr bicudo::gpu::stride_t it_a_vertices {10, 8};
  constexpr bicudo::gpu::stride_t it_a_edges {18, 8};
  constexpr bicudo::gpu::stride_t it_b_vertices {26, 8};
  constexpr bicudo::gpu::stride_t it_b_edges {34, 8};

  struct packed_collision_info_and_two_rect {
  public:
    union {
      struct {
        float32_t has_support_point;
        float32_t best_distance;
        float32_t best_edge;
        float32_t depth;

        float32_t normal_x;
        float32_t normal_y;

        float32_t start_x;
        float32_t start_y;

        float32_t end_x;
        float32_t end_y;

        float32_t a_vertex0_x;
        float32_t a_vertex0_y; 
        float32_t a_vertex1_x;
        float32_t a_vertex1_y;
        float32_t a_vertex2_x;
        float32_t a_vertex2_y;
        float32_t a_vertex3_x;
        float32_t a_vertex3_y;

        float32_t a_edge0_x;
        float32_t a_edge0_y;
        float32_t a_edge1_x;
        float32_t a_edge1_y;
        float32_t a_edge2_x;
        float32_t a_edge2_y;
        float32_t a_edge3_x;
        float32_t a_edge3_y;

        float32_t b_vertex0_x;
        float32_t b_vertex0_y;
        float32_t b_vertex1_x;
        float32_t b_vertex1_y;
        float32_t b_vertex2_x;
        float32_t b_vertex2_y;
        float32_t b_vertex3_x;
        float32_t b_vertex3_y;

        float32_t b_edge0_x;
        float32_t b_edge0_y;
        float32_t b_edge1_x;
        float32_t b_edge1_y;
        float32_t b_edge2_x;
        float32_t b_edge2_y;
        float32_t b_edge3_x;
        float32_t b_edge3_y;
      };

      float32_t host[42] {};
    };
  protected:
    float32_t *p_device {};
    uint64_t buffer_pre_initialized_size {42};
  public:
    inline packed_collision_info_and_two_rect() = default;

    inline float32_t *device_data() {
      return this->p_device;
    }

    inline float32_t *host_data() {
      return this->host;
    }

    inline uint64_t size() {
      return this->buffer_pre_initialized_size;
    };
  }; 
}

#endif