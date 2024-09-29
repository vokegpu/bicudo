#ifndef BICUDO_PHYSICS_PLACEMENT_HPP
#define BICUDO_PHYSICS_PLACEMENT_HPP

#include <vector>
#include "bicudo/util/math.hpp"
#include "types.hpp"

namespace bicudo::physics {
  struct placement {
  public:
    const char *p_tag {};
    bicudo::id id {};

    float mass {};
    float friction {};
    float restitution {};
    float inertia {};

    bicudo::vec2 min {};
    bicudo::vec2 max {};

    bicudo::vec2 pos {};
    bicudo::vec2 size {};
    bicudo::vec2 velocity {};
    bicudo::vec2 acc {};

    float angle {};
    float angular_velocity {};
    float angular_acc {};

    std::vector<bicudo::vec2> vertices {};
    std::vector<bicudo::vec2> edges {};

    bool turn_off_gravity {};
    bool was_collided {};
    bicudo::vec2 prev_size {};
    float prev_mass {};
  };
}

namespace bicudo {
  void physics_placement_move(
    bicudo::physics::placement *p_placement,
    const bicudo::vec2 &dir
  );

  void physics_placement_rotate(
    bicudo::physics::placement *p_placement,
    float angle_dir
  );
}

#endif