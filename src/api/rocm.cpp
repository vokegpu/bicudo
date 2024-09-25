#include "bicudo/api/rocm.hpp"
#include "bicudo/physics/processor.hpp"

void bicudo::api::rocm::init() {
  const char *p_kernel_source_detect_collision {
    R"(

    struct vec2_t {
    public:
      float x {};
      float y {};
    };

    struct stride_t {
    public:
      uint64_t offset {};
      uint64_t size {};
    };

    constexpr stride_t IT_BEST_EDGE {0, 1};
    constexpr stride_t IT_HAS_SUPPORT_POINT {1, 1};
    constexpr stride_t IT_DEPTH {2, 1};
    constexpr stride_t IT_NORMAL {3, 2};
    constexpr stride_t IT_START {5, 2};
    constexpr stride_t IT_END {7, 2};
    constexpr stride_t IT_A_VERTICES {9, 8};
    constexpr stride_t IT_A_EDGES {17, 8};
    constexpr stride_t IT_B_VERTICES {25, 8};
    constexpr stride_t IT_B_EDGES {33, 8};
    constexpr stride_t IT_BEST_DISTANCE {41, 4};
    constexpr stride_t IT_SUPPORT_POINT {45, 4};

    #define CLAMP_MAX(a, b) ((a) > (b) ? (b) : (a))
    #define CLAMP_MIN(a, b) ((a) < (b) ? (b) : (a))
    #define AT(stride, pos) ((p_buffer[stride.offset + pos]))
    #define ATOMIC_READ(at) ((atomicCAS(&at, at, at)))

    extern "C"
    __global__ void detect_collision(
      float *__restrict__ p_buffer
    ) {
      uint32_t index {
        CLAMP_MAX(threadIdx.x, (uint32_t) 3)
      };

      vec2_t edge {
        AT(IT_A_EDGES, (index * 2) + 0),
        AT(IT_A_EDGES, (index * 2) + 1)
      };

      vec2_t dir {
        edge.x * -1.0f,
        edge.y * -1.0f
      };

      vec2_t vert {
        AT(IT_A_VERTICES, (index * 2) + 0),
        AT(IT_A_VERTICES, (index * 2) + 1)
      };

      float dist {
        -99999.0f
      };

      vec2_t vertex {};
      vec2_t to_edge {};
      int32_t best_edge_index_found {-1};
      float proj_dot {};
      int32_t it {};

      it = 0;
      vertex.x = AT(IT_B_VERTICES, (it * 2) + 0);
      vertex.y = AT(IT_B_VERTICES, (it * 2) + 1);

      to_edge.x = vertex.x - vert.x;
      to_edge.y = vertex.y - vert.y;

      proj_dot = (to_edge.x * dir.x + to_edge.y * dir.y);

      if (proj_dot > 0.0f && proj_dot > dist) {
        best_edge_index_found = it;
        dist = proj_dot;
      }

      it = 1;
      vertex.x = AT(IT_B_VERTICES, (it * 2) + 0);
      vertex.y = AT(IT_B_VERTICES, (it * 2) + 1);

      to_edge.x = vertex.x - vert.x;
      to_edge.y = vertex.y - vert.y;

      proj_dot = (to_edge.x * dir.x + to_edge.y * dir.y);

      if (proj_dot > 0.0f && proj_dot > dist) {
        best_edge_index_found = it;
        dist = proj_dot;
      }

      it = 2;
      vertex.x = AT(IT_B_VERTICES, (it * 2) + 0);
      vertex.y = AT(IT_B_VERTICES, (it * 2) + 1);

      to_edge.x = vertex.x - vert.x;
      to_edge.y = vertex.y - vert.y;

      proj_dot = (to_edge.x * dir.x + to_edge.y * dir.y);

      if (proj_dot > 0.0f && proj_dot > dist) {
        best_edge_index_found = it;
        dist = proj_dot;
      }

      it = 3;
      vertex.x = AT(IT_B_VERTICES, (it * 2) + 0);
      vertex.y = AT(IT_B_VERTICES, (it * 2) + 1);

      to_edge.x = vertex.x - vert.x;
      to_edge.y = vertex.y - vert.y;

      proj_dot = (to_edge.x * dir.x + to_edge.y * dir.y);

      if (proj_dot > 0.0f && proj_dot > dist) {
        best_edge_index_found = it;
        dist = proj_dot;
      }

      if (best_edge_index_found != -1) {
        //AT(IT_BEST_DISTANCE, index) = dist;
        //AT(IT_SUPPORT_POINT, index) = best_edge_index_found;
        //AT(IT_HAS_SUPPORT_POINT, 0) = 1.0f;
        atomicExch(&AT(IT_BEST_DISTANCE, index), dist);
        atomicExch(&AT(IT_SUPPORT_POINT, index), best_edge_index_found);
        atomicAdd(&AT(IT_HAS_SUPPORT_POINT, 0), 1.0f);
      }

      if ((int32_t) AT(IT_HAS_SUPPORT_POINT, 0) == 4) {
        int32_t best_edge_index {};
        int32_t support_point_index {};
        dist = 99999.0f;

        if (AT(IT_BEST_DISTANCE, 0) < dist) {
          best_edge_index = 0;
          support_point_index = AT(IT_SUPPORT_POINT, 0);
          dist = AT(IT_BEST_DISTANCE, 0);
        }

        if (AT(IT_BEST_DISTANCE, 1) < dist) {
          best_edge_index = 1;
          support_point_index = AT(IT_SUPPORT_POINT, 1);
          dist = AT(IT_BEST_DISTANCE, 1);
        }

        if (AT(IT_BEST_DISTANCE, 2) < dist) {
          best_edge_index = 2;
          support_point_index = AT(IT_SUPPORT_POINT, 2);
          dist = AT(IT_BEST_DISTANCE, 2);
        }

        if (AT(IT_BEST_DISTANCE, 3) < dist) {
          best_edge_index = 3;
          support_point_index = AT(IT_SUPPORT_POINT, 3);
          dist = AT(IT_BEST_DISTANCE, 3);
        }

        vec2_t best_edge {
          AT(IT_A_EDGES, (best_edge_index * 2) + 0),
          AT(IT_A_EDGES, (best_edge_index * 2) + 1)
        };

        float best_distance {
          dist
        };

        AT(IT_DEPTH, 0) = best_distance;
        AT(IT_BEST_EDGE, 0) = best_edge_index;
        AT(IT_NORMAL, 0) = best_edge.x;
        AT(IT_NORMAL, 1) = best_edge.y;

        vec2_t support_point {
          AT(IT_B_VERTICES, (support_point_index * 2) + 0),
          AT(IT_B_VERTICES, (support_point_index * 2) + 1)
        };

        AT(IT_START, 0) = support_point.x + (best_edge.x * best_distance);
        AT(IT_START, 1) = support_point.y + (best_edge.y * best_distance);

        AT(IT_END, 0) = AT(IT_START, 0) + best_edge.x * best_distance;
        AT(IT_END, 1) = AT(IT_START, 1) + best_edge.y * best_distance;
      }
    }

    )"
  };

  uint64_t float32_size {
    sizeof(float32_t)
  };

  bicudo::gpu::rocm_pipeline_create_info rocm_pipeline_create_info {
    .p_tag = "world-physics-pipeline",
    .kernel_list = {
      {
        .p_tag = "detect-collision",
        .p_src = p_kernel_source_detect_collision,
        .function_list = {
          {
            .p_entry_point = "detect_collision",
            .grid = dim3(1, 1, 1),
            .block = dim3(4, 1, 1),
            .shared_mem_bytes = 0,
            .stream = nullptr,
            .buffer_list = {
              {
                .size = (
                  this->detect_collision_memory.size()
                  *
                  float32_size
                ),
                .p_device = this->detect_collision_memory.device_data(),
                .p_host = this->detect_collision_memory.host_data()
              }
            }
          }
        }
      }
    }
  };

  bicudo::result result {
    bicudo::gpu_rocm_create_pipeline(
      &this->pipeline,
      &rocm_pipeline_create_info
    )
  };

  this->detect_collision_memory_index = 0;

  if (result == bicudo::FAILED) {
    bicudo::log() << "Failed to world physics service compile the following pipeline: " << this->pipeline.p_tag;
  }
}

void bicudo::api::rocm::quit() {
  
}

void bicudo::api::rocm::update_physics_simulator(
  bicudo::physics::placement *&p_a,
  bicudo::physics::placement *&p_b,
  bicudo::physics::collision_info_t *p_collision_info
) {
  p_collision_info->collided = false;

  bicudo::physics::collision_info_t a_collision_info {};
  this->compute_detect_collision_kernel(
    &a_collision_info,
    p_a,
    p_b
  );

  if (!a_collision_info.has_support_point) {
    return;
  }
    
  bicudo::physics::collision_info_t b_collision_info {};
  this->compute_detect_collision_kernel(
    &b_collision_info,
    p_b,
    p_a
  );

  if (!b_collision_info.has_support_point) {
    return;
  }
    
  if (a_collision_info.depth < b_collision_info.depth) {
    p_collision_info->depth = a_collision_info.depth;
    p_collision_info->normal = a_collision_info.normal;
    p_collision_info->start = a_collision_info.start - (a_collision_info.normal * a_collision_info.depth);
    
    bicudo::physics_processor_collision_info_update(
      p_collision_info
    );
  } else {
    p_collision_info->depth = b_collision_info.depth;
    p_collision_info->normal = b_collision_info.normal * -1.0f;
    p_collision_info->start = b_collision_info.start;
    
    bicudo::physics_processor_collision_info_update(
      p_collision_info
    );
  }

  p_collision_info->collided = true;
}

void bicudo::api::rocm::compute_detect_collision_kernel(
  bicudo::physics::collision_info_t *p_collision_info,
  bicudo::physics::placement *&p_a,
  bicudo::physics::placement *&p_b
) {

  /**
   * First reset the host memory collision info,
   * then write-store both rects a and b to host device memory. 
   **/

  this->collision_info_memory_fetch(
    &this->detect_collision_memory,
    p_collision_info,
    bicudo::types::WRITEBACK
  );

  this->rect_a_b_memory_writestore(
    &this->detect_collision_memory,
    p_a,
    p_b
  );

  /**
   * Fetch the host memory to the GPU device,
   * storing all fields in buffer content.
   **/

  bicudo::gpu_rocm_memory_fetch(
    &this->pipeline,
    0,
    0,
    this->detect_collision_memory_index,
    bicudo::types::WRITESTORE
  );

  /**
   * Compute the kernel and get the results.
   **/

  bicudo::gpu_rocm_dispatch(
    &this->pipeline,
    0,
    0
  );

  /**
   * Fetch the GPU device memory to the host,
   * write-back the collision info final results
   * and collision info.
   **/

  bicudo::gpu_rocm_memory_fetch(
    &this->pipeline,
    0,
    0,
    this->detect_collision_memory_index,
    bicudo::types::WRITEBACK
  );

  this->collision_info_memory_fetch(
    &this->detect_collision_memory,
    p_collision_info,
    bicudo::types::WRITESTORE
  );
}

void bicudo::api::rocm::collision_info_memory_fetch(
  bicudo::gpu::packed_collision_info_and_two_rect *p_packed,
  bicudo::physics::collision_info_t *p_collision_info,
  bicudo::types op_type
) {
  switch (op_type) {
  case bicudo::types::WRITEBACK:
    p_collision_info->has_support_point = false;
    p_packed->has_support_point = static_cast<float>(p_collision_info->has_support_point);
    p_packed->best_distance0 = 99999.0f;
    p_packed->best_distance1 = 99999.0f;
    p_packed->best_distance2 = 99999.0f;
    p_packed->best_distance3 = 99999.0f;
    p_packed->support_point0 = 0;
    p_packed->support_point1 = 0;
    p_packed->support_point2 = 0;
    p_packed->support_point3 = 0;
    p_packed->best_edge = 0.0f;
    break;
  case bicudo::types::WRITESTORE:
    p_collision_info->has_support_point = static_cast<int32_t>(p_packed->has_support_point) > 3;
    p_collision_info->depth = p_packed->depth;
    p_collision_info->normal.x = p_packed->normal_x;
    p_collision_info->normal.y = p_packed->normal_y;
    p_collision_info->start.x = p_packed->start_x;
    p_collision_info->start.y = p_packed->start_y;
    p_collision_info->end.x = p_packed->end_x;
    p_collision_info->end.y = p_packed->end_y;
    break;
  }
}

void bicudo::api::rocm::rect_a_b_memory_writestore(
  bicudo::gpu::packed_collision_info_and_two_rect *p_packed,
  bicudo::physics::placement *&p_a,
  bicudo::physics::placement *&p_b
) {
  p_packed->a_vertex0_x = p_a->vertices.at(0).x;
  p_packed->a_vertex0_y = p_a->vertices.at(0).y;
  p_packed->a_vertex1_x = p_a->vertices.at(1).x;
  p_packed->a_vertex1_y = p_a->vertices.at(1).y;
  p_packed->a_vertex2_x = p_a->vertices.at(2).x;
  p_packed->a_vertex2_y = p_a->vertices.at(2).y;
  p_packed->a_vertex3_x = p_a->vertices.at(3).x;
  p_packed->a_vertex3_y = p_a->vertices.at(3).y;

  p_packed->a_edge0_x = p_a->edges.at(0).x;
  p_packed->a_edge0_y = p_a->edges.at(0).y;
  p_packed->a_edge1_x = p_a->edges.at(1).x;
  p_packed->a_edge1_y = p_a->edges.at(1).y;
  p_packed->a_edge2_x = p_a->edges.at(2).x;
  p_packed->a_edge2_y = p_a->edges.at(2).y;
  p_packed->a_edge3_x = p_a->edges.at(3).x;
  p_packed->a_edge3_y = p_a->edges.at(3).y;

  p_packed->b_vertex0_x = p_b->vertices.at(0).x;
  p_packed->b_vertex0_y = p_b->vertices.at(0).y;
  p_packed->b_vertex1_x = p_b->vertices.at(1).x;
  p_packed->b_vertex1_y = p_b->vertices.at(1).y;
  p_packed->b_vertex2_x = p_b->vertices.at(2).x;
  p_packed->b_vertex2_y = p_b->vertices.at(2).y;
  p_packed->b_vertex3_x = p_b->vertices.at(3).x;
  p_packed->b_vertex3_y = p_b->vertices.at(3).y;

  p_packed->b_edge0_x = p_b->edges.at(0).x;
  p_packed->b_edge0_y = p_b->edges.at(0).y;
  p_packed->b_edge1_x = p_b->edges.at(1).x;
  p_packed->b_edge1_y = p_b->edges.at(1).y;
  p_packed->b_edge2_x = p_b->edges.at(2).x;
  p_packed->b_edge2_y = p_b->edges.at(2).y;
  p_packed->b_edge3_x = p_b->edges.at(3).x;
  p_packed->b_edge3_y = p_b->edges.at(3).y;
}