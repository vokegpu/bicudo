#ifndef BICUDO_PHYSICS_PROCESSOR_HPP
#define BICUDO_PHYSICS_PROCESSOR_HPP

#include "bicudo/bicudo.hpp"

namespace bicudo {
  void physics_processor_update(
    bicudo::runtime *p_runtime
  );
  
  bicudo::collided physics_processor_a_collide_with_b_check(
    bicudo::physics::placement *&p_a,
    bicudo::physics::placement *&p_b,
    bicudo::runtime *p_runtime
  );

  void physics_processor_collision_info_change_dir(
    bicudo::physics::collision_info_t *p_collision_info
  );

  void physics_processor_collision_info_update(
    bicudo::physics::collision_info_t *p_collsion_info
  );

  void physics_processor_find_axis_penetration(
    bicudo::physics::placement *&p_a,
    bicudo::physics::placement *&p_b,
    bicudo::physics::collision_info_t *p_collision_info
  );
}

#endif