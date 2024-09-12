#ifndef BICUDO_WORLD_PHYSICS_SIMULATOR_HPP
#define BICUDO_WORLD_PHYSICS_SIMULATOR_HPP

#include "bicudo/util/math.hpp"
#include "bicudo/world/types.hpp"
#include <vector>

namespace bicudo::world::physics {
  struct simulator {
  public:
    std::vector<bicudo::placement*> placement_list {};
  };

  struct collision_info_t {
  public:
    float depth {};
    bicudo::vec2 normal {};
    bicudo::vec2 start {};
    bicudo::vec2 end {};
  };

  struct support_info_t {
  public:
    bicudo::vec2 point {};
    float projection {};
    float distance {};
  };
}

namespace bicudo {
  void world_physics_update_simulator(
    bicudo::world::physics::simulator *p_simulator
  );
  
  bicudo::collided world_physics_a_collide_with_b_check(
    bicudo::placement *&p_a,
    bicudo::placement *&p_b,
    bicudo::world::physics::collision_info_t *p_collision_info,
    bicudo::world::physics::support_info_t *p_support_info
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
    bicudo::world::physics::support_info_t *p_support_info,
    bool *p_has_support_point
  );
}

#endif
