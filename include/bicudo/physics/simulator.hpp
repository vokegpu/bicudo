#ifndef BICUDO_PHYSICS_SIMULATOR_HPP
#define BICUDO_PHYSICS_SIMULATOR_HPP

#include "bicudo/util/math.hpp"
#include "types.hpp"
#include "placement.hpp"
#include "bicudo/gpu/model.hpp"
#include "bicudo/gpu/algebra_linear.hpp"

namespace bicudo::physics {
  struct collision_info_t {
  public:
    float depth {};
    bicudo::vec2 normal {};
    bicudo::vec2 start {};
    bicudo::vec2 end {};
    bicudo::collided has_support_point {};
  };

  struct simulator {
  public:
    bicudo::gpu::pipeline pipeline {};
    bicudo::gpu::packed_collision_info_and_two_rect detect_collision_memory {};
    uint64_t detect_collision_memory_index {};
  public:
    bicudo::physics_runtime_type physics_runtime_type {};
    std::vector<bicudo::physics::placement*> placement_list {};
    bicudo::physics::collision_info_t collision_info {};
  };
}

namespace bicudo {
  void physics_compute_detect_collision_kernel(
    bicudo::physics::simulator *p_simulator,
    bicudo::physics::collision_info_t *p_collision_info,
    bicudo::physics::placement *&p_a,
    bicudo::physics::placement *&p_b
  );

  void physics_collision_info_memory_fetch(
    bicudo::gpu::packed_collision_info_and_two_rect *p_packed,
    bicudo::physics::collision_info_t *p_collision_info,
    bicudo::types op_type
  );

  void physics_rect_a_b_memory_writestore(
    bicudo::gpu::packed_collision_info_and_two_rect *p_packed,
    bicudo::physics::placement *&p_a,
    bicudo::physics::placement *&p_b
  );

  void physics_init(
    bicudo::physics::simulator *p_simulator
  );

  void physics_update_simulator(
    bicudo::physics::simulator *p_simulator
  );
  
  bicudo::collided physics_a_collide_with_b_check(
    bicudo::physics::placement *&p_a,
    bicudo::physics::placement *&p_b,
    bicudo::physics::collision_info_t *p_collision_info,
    bicudo::physics::simulator *p_simulator
  );

  void physics_collision_info_change_dir(
    bicudo::physics::collision_info_t *p_collision_info
  );

  void physics_collision_info_update(
    bicudo::physics::collision_info_t *p_collsion_info
  );

  void physics_find_axis_penetration(
    bicudo::physics::placement *&p_a,
    bicudo::physics::placement *&p_b,
    bicudo::physics::collision_info_t *p_collision_info
  );
}

#endif