#ifndef BICUDO_WORLD_PHYSICS_SIMULATOR_HPP
#define BICUDO_WORLD_PHYSICS_SIMULATOR_HPP

#include "bicudo/util/math.hpp"
#include "bicudo/world/types.hpp"
#include "bicudo/gpu/model.hpp"
#include "bicudo/gpu/algebra_linear.hpp"
#include <vector>

namespace bicudo::world::physics {
  struct collision_info_t {
  public:
    float depth {};
    bicudo::vec2 normal {};
    bicudo::vec2 start {};
    bicudo::vec2 end {};
  };

  struct detect_collision_memory {
  public:
    bicudo::gpu::rect_t rect_a {};
    bicudo::gpu::rect_t rect_b {};
    bicudo::gpu::collision_info_t collision_info {};
  };

  struct simulator {
  public:
    bicudo::gpu::pipeline pipeline {};
    bicudo::world::physics::detect_collision_memory host_detect_collision_memory {};
    bicudo::world::physics::detect_collision_memory device_detect_collision_memory {};
  public:
    std::vector<bicudo::placement*> placement_list {};
    bicudo::world::physics::collision_info_t collision_info {};
  };
}

namespace bicudo {

  void world_physics_init(
    bicudo::world::physics::simulator *p_simulator
  );

  void world_physics_update_simulator(
    bicudo::world::physics::simulator *p_simulator
  );
  
  bicudo::collided world_physics_a_collide_with_b_check(
    bicudo::placement *&p_a,
    bicudo::placement *&p_b,
    bicudo::world::physics::collision_info_t *p_collision_info
  );

  void world_physics_collision_info_change_dir(
    bicudo::world::physics::collision_info_t *p_collision_info
  );

  void world_physics_collision_info_update(
    bicudo::world::physics::collision_info_t *p_collsion_info
  );

  void world_physics_find_axis_penetration(
    bicudo::placement *&p_a,
    bicudo::placement *&p_b,
    bicudo::world::physics::collision_info_t *p_collision_info,
    bool *p_has_support_point
  );
}

#endif
