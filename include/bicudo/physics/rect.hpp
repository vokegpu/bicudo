#ifndef BICUDO_PHYSICS_RECT_HPP
#define BICUDO_PHYSICS_RECT_HPP

#include <bicudo/math/geometry.hpp>

namespace bicudo {
  template<typename t>
  struct rect_t {
  public:
    bicudo::vec2_t<t> pos {};
    bicudo::vec2_t<t> size {};
    bicudo::vec2_t<t> velocity {};
    bicudo::vec2_t<t> acceleration {};

    t angle {};
    t angle_velocity {};
    t angle_acceleration {};
  };
}

#endif
