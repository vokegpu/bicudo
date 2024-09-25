#ifndef BICUDO_API_BASE_HPP
#define BICUDO_API_BASE_HPP

#include "bicudo/physics/placement.hpp"
#include "bicudo/physics/collision_info.hpp"

namespace bicudo::api {
  struct base {
  public:
    virtual void init() {};
    virtual void quit() {};

    virtual void update_physics_simulator(
      bicudo::physics::placement *&p_a,
      bicudo::physics::placement *&p_b,
      bicudo::physics::collision_info_t *p_collision_info
    ) {};
  };
}

#endif