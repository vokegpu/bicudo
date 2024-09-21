#include "bicudo/world/physics/simulator.hpp"
#include "bicudo/util/log.hpp"
#include "bicudo/bicudo.hpp"

void bicudo::world_physics_compute_detect_collision_kernel(
  bicudo::world::physics::simulator *p_simulator,
  bicudo::world::physics::collision_info_t *p_collision_info,
  bicudo::placement *&p_a,
  bicudo::placement *&p_b
) {

  /**
   * First reset the host memory collision info,
   * then write-store both rects a and b to host device memory. 
   **/

  bicudo::world_physics_collision_info_memory_fetch(
    &p_simulator->detect_collision_memory,
    p_collision_info,
    bicudo::types::WRITEBACK
  );

  bicudo::world_physics_rect_a_b_memory_writestore(
    &p_simulator->detect_collision_memory,
    p_a,
    p_b
  );

  /**
   * Fetch the host memory to the GPU device,
   * storing all fields in buffer content.
   **/

  bicudo::gpu_memory_fetch(
    &p_simulator->pipeline,
    0,
    0,
    p_simulator->detect_collision_memory_index,
    bicudo::types::WRITESTORE
  );

  /**
   * Compute the kernel and get the results.
   **/

  bicudo::gpu_dispatch(
    &p_simulator->pipeline,
    0,
    0
  );

  /**
   * Fetch the GPU device memory to the host,
   * write-back the collision info final results
   * and collision info.
   **/

  bicudo::gpu_memory_fetch(
    &p_simulator->pipeline,
    0,
    0,
    p_simulator->detect_collision_memory_index,
    bicudo::types::WRITEBACK
  );

  bicudo::world_physics_collision_info_memory_fetch(
    &p_simulator->detect_collision_memory,
    p_collision_info,
    bicudo::types::WRITESTORE
  );
}

void bicudo::world_physics_collision_info_memory_fetch(
  bicudo::gpu::packed_collision_info_and_two_rect *p_packed,
  bicudo::world::physics::collision_info_t *p_collision_info,
  bicudo::types op_type
) {
  switch (op_type) {
  case bicudo::types::WRITEBACK:
    p_collision_info->has_support_point = false;
    p_packed->has_support_point = static_cast<float>(p_collision_info->has_support_point);
    p_packed->best_distance = 99999.0f;
    p_packed->best_edge = 0.0f;
    break;
  case bicudo::types::WRITESTORE:
    p_collision_info->has_support_point = static_cast<bool>(p_packed->has_support_point);
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

void bicudo::world_physics_rect_a_b_memory_writestore(
  bicudo::gpu::packed_collision_info_and_two_rect *p_packed,
  bicudo::placement *&p_a,
  bicudo::placement *&p_b
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

void bicudo::world_physics_init(
  bicudo::world::physics::simulator *p_simulator
) {
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

    constexpr stride_t IT_BEST_DISTANCE {0, 1};
    constexpr stride_t IT_BEST_EDGE {1, 2};
    constexpr stride_t IT_HAS_SUPPORT_POINT {2, 1};
    constexpr stride_t IT_DEPTH {3, 1};
    constexpr stride_t IT_NORMAL {4, 2};
    constexpr stride_t IT_START {6, 2};
    constexpr stride_t IT_END {8, 2};
    constexpr stride_t IT_A_VERTICES {10, 8};
    constexpr stride_t IT_A_EDGES {18, 8};
    constexpr stride_t IT_B_VERTICES {26, 8};
    constexpr stride_t IT_B_EDGES {34, 8};

    #define CLAMP_MAX(a, b)  ((a) > (b) ? (b) : (a))
    #define AT(stride, pos) ((p_buffer[stride.offset + pos]))

    extern "C"
    __global__ void detect_collision(
      float *p_buffer
    ) {
      int32_t index {
        CLAMP_MAX(threadIdx.x, 4)
      };

      if ((int32_t) AT(IT_HAS_SUPPORT_POINT, 0) == 4) {
        int32_t best_edge_index {
          (int32_t) AT(IT_BEST_EDGE, 0)
        };

        vec2_t best_edge {
          AT(IT_A_EDGES, (best_edge_index * 2) + 0),
          AT(IT_A_EDGES, (best_edge_index * 2) + 1)
        };

        float best_distance {
          AT(IT_BEST_DISTANCE, 0)
        };

        AT(IT_DEPTH, 0) = best_distance;
        AT(IT_NORMAL, 0) = best_edge.x;
        AT(IT_NORMAL, 1) = best_edge.y;

        vec2_t support_point {
          AT(IT_START, 0),
          AT(IT_START, 1)
        };

        AT(IT_START, 0) = support_point.x + (best_edge.x * best_distance);
        AT(IT_START, 1) = support_point.y + (best_edge.y * best_distance);

        AT(IT_END, 0) = AT(IT_START, 0) + AT(IT_NORMAL, 0) * best_distance;
        AT(IT_END, 1) = AT(IT_START, 1) + AT(IT_NORMAL, 1) * best_distance;

        return;
      } else if ((int32_t) AT(IT_HAS_SUPPORT_POINT, 0) == -1) {
        return;
      }

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
      float proj {};
      vec2_t point {};
      float has_support_point {};
      
      for (int32_t it {}; it < 4; it++) {
        vertex.x = AT(IT_B_VERTICES, (it * 2) + 0);
        vertex.y = AT(IT_B_VERTICES, (it * 2) + 1);

        to_edge.x = vertex.x - vert.x;
        to_edge.y = vertex.y - vert.y;

        proj = (to_edge.x * dir.x + to_edge.y * dir.y);

        if (proj > 0.0f && proj > dist) {
          point = vertex;
          dist = proj;
          has_support_point = 1.0f;
        }
      }

      if (has_support_point > 0.0f && dist < AT(IT_BEST_DISTANCE, 0)) {
        AT(IT_BEST_DISTANCE, 0) = dist;
        AT(IT_BEST_EDGE, 0) = index;
        AT(IT_START, 0) = point.x;
        AT(IT_START, 1) = point.y;
        AT(IT_HAS_SUPPORT_POINT, 0) += 1.0f;
      } else {
        AT(IT_HAS_SUPPORT_POINT, 0) = -1.0f;
      }
    }

    )"
  };

  uint64_t float32_size {
    sizeof(float32_t)
  };

  bicudo::gpu::pipeline_create_info pipeline_create_info {
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
                  p_simulator->detect_collision_memory.size()
                  *
                  float32_size
                ),
                .p_device = p_simulator->detect_collision_memory.device_data(),
                .p_host = p_simulator->detect_collision_memory.host_data()
              }
            }
          }
        }
      }
    }
  };

  bicudo::result result {
    bicudo::gpu_create_pipeline(
      &p_simulator->pipeline,
      &pipeline_create_info
    )
  };

  p_simulator->detect_collision_memory_index = 0;

  if (result == bicudo::FAILED) {
    bicudo::log() << "Failed to world physics service compile the following pipeline: " << p_simulator->pipeline.p_tag;
  }

  p_simulator->detect_collision_memory.best_distance = 99999.0f;
  p_simulator->detect_collision_memory.best_edge = 0.0f;
  p_simulator->detect_collision_memory.has_support_point = 0.0f;

  bicudo::gpu_memory_fetch(
    &p_simulator->pipeline,
    0,
    0,
    0,
    bicudo::types::WRITESTORE
  );

  bicudo::gpu_dispatch(
    &p_simulator->pipeline,
    0,
    0
  );

  //bicudo::log() << "Before writeback:";

  float32_t *p_buffer {p_simulator->detect_collision_memory.host_data()};
  for (uint64_t it {}; it < p_simulator->detect_collision_memory.size(); it++) {
    //bicudo::log() << p_buffer[it];
  }

  bicudo::gpu_memory_fetch(
    &p_simulator->pipeline,
    0,
    0,
    0,
    bicudo::types::WRITEBACK
  );

  //bicudo::log() << "Post writeback:";

  for (uint64_t it {}; it < p_simulator->detect_collision_memory.size(); it++) {
    //bicudo::log() << p_buffer[it];
  }
}

void bicudo::world_physics_update_simulator(
  bicudo::world::physics::simulator *p_simulator
) {
  bicudo::collided was_collided {};

  float num {};
  bicudo::vec2 correction {};

  bicudo::vec2 n {};
  bicudo::vec2 start {};
  bicudo::vec2 end {};
  bicudo::vec2 p {};
  float total_mass {};

  bicudo::vec2 c1 {};
  bicudo::vec2 c2 {};

  float c1_cross {};
  float c2_cross {};

  bicudo::vec2 v1 {};
  bicudo::vec2 v2 {};
  bicudo::vec2 vdiff {};
  float vdiff_dot {};

  float restitution {};
  float friction {};
  float jn {};
  float jt {};

  bicudo::vec2 impulse {};
  bicudo::vec2 tangent {};

  uint64_t placement_size {p_simulator->placement_list.size()};

  switch (bicudo::app.physics_runtime_type) {
  case bicudo::physics_runtime_type::CPU_SIDE:
    for (uint64_t it_a {}; it_a < placement_size; it_a++) {
      /* stupid */
      for (uint64_t it_b {}; it_b < placement_size; it_b++) {
        if (it_a == it_b) {
          continue;
        }
  
        bicudo::placement *&p_a {p_simulator->placement_list.at(it_a)};
        bicudo::placement *&p_b {p_simulator->placement_list.at(it_b)};
  
        if (bicudo::assert_float(p_a->mass, 0.0f) && bicudo::assert_float(p_b->mass, 0.0f)) {
          continue;
        }
  
        p_simulator->collision_info = {};
  
        was_collided = (
          bicudo::world_physics_a_collide_with_b_check(
            p_a,
            p_b,
            &p_simulator->collision_info,
            p_simulator
          )
        );
  
        p_a->was_collided = was_collided;
  
        if (!was_collided) {
          continue;
        }
  
        n = p_simulator->collision_info.normal;
        total_mass = p_a->mass + p_b->mass;
        num = p_simulator->collision_info.depth / total_mass * 1.0f;
        correction = n * num;
  
        bicudo::move(
          p_a,
          correction * -p_a->mass
        );
  
        bicudo::move(
          p_b,
          correction * p_b->mass
        );
  
        start = p_simulator->collision_info.start * (p_b->mass / total_mass);
        end = p_simulator->collision_info.end * (p_a->mass / total_mass);
        p = start + end;
  
        c1.x = p_a->pos.x + (p_a->size.x / 2);
        c1.y = p_a->pos.y + (p_a->size.y / 2);
        c1 = p - c1;
      
        c2.x = p_b->pos.x + (p_b->size.x / 2);
        c2.y = p_b->pos.y + (p_b->size.y / 2);
        c2 = p - c2;
  
        v1 = (
          p_a->velocity + bicudo::vec2(-1.0f * p_a->angular_velocity * c1.y, p_a->angular_velocity * c1.x)
        );
  
        v2 = (
          p_b->velocity + bicudo::vec2(-1.0f * p_b->angular_velocity * c2.y, p_b->angular_velocity * c2.x)
        );
  
        vdiff = v2 - v1;
        vdiff_dot = vdiff.dot(n);
  
        if (vdiff_dot > 0.0f) {
          continue;
        }
  
        restitution = std::min(p_a->restitution, p_b->restitution);
        friction = std::min(p_a->friction, p_b->friction);
      
        c1_cross = c1.cross(n);
        c2_cross = c2.cross(n);
  
        jn = (
          (-(1.0f + restitution) * vdiff_dot)
          /
          (total_mass + c1_cross * c1_cross * p_a->inertia + c2_cross * c2_cross * p_b->inertia)
        );
  
        impulse = n * jn;
  
        p_a->velocity -= impulse * p_a->mass;
        p_b->velocity += impulse * p_b->mass;
  
        p_a->angular_velocity -= c1_cross * jn * p_a->inertia;
        p_b->angular_velocity += c2_cross * jn * p_b->inertia;
  
        tangent = vdiff - n * vdiff.dot(n);
        tangent = tangent.normalize() * -1.0f;
  
        c1_cross = c1.cross(tangent);
        c2_cross = c2.cross(tangent);
  
        jt = (
          (-(1.0f + restitution) * vdiff.dot(tangent) * friction)
          /
          (total_mass + c1_cross * c1_cross * p_a->inertia + c2_cross * c2_cross * p_b->inertia)
        );
  
        jt = jt > jn ? jn : jt;
        impulse = tangent * jt;
  
        p_a->velocity -= impulse * p_a->mass;
        p_b->velocity += impulse * p_b->mass;
  
        p_a->angular_velocity -= c1_cross * jt * p_a->inertia;
        p_b->angular_velocity += c2_cross * jt * p_b->inertia;
      }
    }
    break;
  case bicudo::physics_runtime_type::GPU_ROCM:
    for (uint64_t it_a {}; it_a < placement_size; it_a++) {
      /* stupid */
      for (uint64_t it_b {}; it_b < placement_size; it_b++) {
        if (it_a == it_b) {
          continue;
        }
  
        bicudo::placement *&p_a {p_simulator->placement_list.at(it_a)};
        bicudo::placement *&p_b {p_simulator->placement_list.at(it_b)};
  
        if (bicudo::assert_float(p_a->mass, 0.0f) && bicudo::assert_float(p_b->mass, 0.0f)) {
          continue;
        }
  
        p_simulator->collision_info = {};
  
        was_collided = (
          bicudo::world_physics_a_collide_with_b_check(
            p_a,
            p_b,
            &p_simulator->collision_info,
            p_simulator
          )
        );
  
        p_a->was_collided = was_collided;
      }
    }

    break;
  }
}

bicudo::collided bicudo::world_physics_a_collide_with_b_check(
  bicudo::placement *&p_a,
  bicudo::placement *&p_b,
  bicudo::world::physics::collision_info_t *p_collision_info,
  bicudo::world::physics::simulator *p_simulator
) {
  switch (bicudo::app.physics_runtime_type) {
    case bicudo::physics_runtime_type::CPU_SIDE: {
      bicudo::world::physics::collision_info_t a_collision_info {};
      bicudo::world_physics_find_axis_penetration(
        p_a,
        p_b,
        &a_collision_info
      );

      if (!a_collision_info.has_support_point) {
        return false;
      }

      bicudo::world::physics::collision_info_t b_collision_info {};
      bicudo::world_physics_find_axis_penetration(
        p_b,
        p_a,
        &b_collision_info
      );
    
      if (!b_collision_info.has_support_point) {
        return false;
      }
    
      if (a_collision_info.depth < b_collision_info.depth) {
        p_collision_info->depth = a_collision_info.depth;
        p_collision_info->normal = a_collision_info.normal;
        p_collision_info->start = a_collision_info.start - (a_collision_info.normal * a_collision_info.depth);
    
        bicudo::world_physics_collision_info_update(
          p_collision_info
        );
      } else {
        p_collision_info->depth = b_collision_info.depth;
        p_collision_info->normal = b_collision_info.normal * -1.0f;
        p_collision_info->start = b_collision_info.start;
    
        bicudo::world_physics_collision_info_update(
          p_collision_info
        );
      }
    
      return true;
    }

    case bicudo::physics_runtime_type::GPU_ROCM: {
      bicudo::world::physics::collision_info_t a_collision_info {};
      bicudo::world_physics_compute_detect_collision_kernel(
        p_simulator,
        &a_collision_info,
        p_a,
        p_b
      );

      if (!a_collision_info.has_support_point) {
        return false;
      }
    
      bicudo::world::physics::collision_info_t b_collision_info {};
      bicudo::world_physics_compute_detect_collision_kernel(
        p_simulator,
        &b_collision_info,
        p_b,
        p_a
      );

      if (!b_collision_info.has_support_point) {
        return false;
      }
    
      if (a_collision_info.depth < b_collision_info.depth) {
        p_collision_info->depth = a_collision_info.depth;
        p_collision_info->normal = a_collision_info.normal;
        p_collision_info->start = a_collision_info.start - (a_collision_info.normal * a_collision_info.depth);
    
        bicudo::world_physics_collision_info_update(
          p_collision_info
        );
      } else {
        p_collision_info->depth = b_collision_info.depth;
        p_collision_info->normal = b_collision_info.normal * -1.0f;
        p_collision_info->start = b_collision_info.start;
    
        bicudo::world_physics_collision_info_update(
          p_collision_info
        );
      }
    
      return true;
    }
  }

  return false;
}

void bicudo::world_physics_collision_info_change_dir(
  bicudo::world::physics::collision_info_t *p_collision_info
) {
  p_collision_info->normal *= -1.0f;
  bicudo::vec2 n {p_collision_info->normal};
  p_collision_info->start = p_collision_info->end;
  p_collision_info->end = n;
}

void bicudo::world_physics_collision_info_update(
  bicudo::world::physics::collision_info_t *p_collsion_info
) {
  p_collsion_info->end = (
    p_collsion_info->start + p_collsion_info->normal * p_collsion_info->depth
  );
}

void bicudo::world_physics_find_axis_penetration(
  bicudo::placement *&p_a,
  bicudo::placement *&p_b,
  bicudo::world::physics::collision_info_t *p_collision_info
) {
  bicudo::vec2 edge {};
  bicudo::vec2 support_point {};

  float best_dist {FLT_MAX};
  uint64_t best_edge {};
  p_collision_info->has_support_point = true;

  bicudo::vec2 dir {};
  bicudo::vec2 vert {};
  bicudo::vec2 to_edge {};

  float proj {};
  float dist {};

  bicudo::vec2 point {};

  uint64_t edges_size {p_a->edges.size()};
  uint64_t vertices_size {p_b->vertices.size()};

  for (uint64_t it_edges {}; p_collision_info->has_support_point && it_edges < edges_size; it_edges++) {
    edge = p_a->edges.at(it_edges); // normalized edge
    dir = edge * -1.0f;
    vert = p_a->vertices.at(it_edges);

    dist = -FLT_MAX;
    p_collision_info->has_support_point = false;

    for (bicudo::vec2 &vertex : p_b->vertices) {
      to_edge = vertex - vert;
      proj = to_edge.dot(dir);

      if (proj > 0.0f && proj > dist) {
        point = vertex;
        dist = proj;
        p_collision_info->has_support_point = true;
      }
    }

    if (p_collision_info->has_support_point && dist < best_dist) {
      best_dist = dist;
      best_edge = it_edges;
      support_point = point;
    }
  }

  if (p_collision_info->has_support_point) {
    edge = p_a->edges.at(best_edge);

    p_collision_info->depth = best_dist;
    p_collision_info->normal = edge;
    p_collision_info->start = support_point + (edge * best_dist);

    bicudo::world_physics_collision_info_update(
      p_collision_info
    );
  }
}
