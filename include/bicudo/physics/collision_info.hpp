#ifndef BICUDO_PHYSICS_COLLISION_INFO_HPP
#define BICUDO_PHYSICS_COLLISION_INFO_HPP

#include "placement.hpp"

namespace bicudo::physics {
  struct collision_info_t {
  public:
    float depth {};
    bicudo::vec2 normal {};
    bicudo::vec2 start {};
    bicudo::vec2 end {};
    bicudo::collided has_support_point {};
    bicudo::collided collided {};
  };
}

#endif