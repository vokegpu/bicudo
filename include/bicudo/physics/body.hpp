#ifndef BICUDO_PHYSICS_BODY_HPP
#define BICUDO_PHYSICS_BODY_HPP

#include <bicudo/math/geometry.hpp>
#include <bicudo/io/signature.hpp>
#include <vector>

namespace bicudo {
  struct body_t {
  public:
    bicudo::vec2_t<float> pos {};
    bicudo::vec2_t<float> size {};
    bicudo::vec2_t<float> velocity {};
    bicudo::vec2_t<float> acceleration {};

    std::vector<bicudo::vec2_t<float>> edges {};
    std::vector<bicudo::vec2_t<float>> vertices {};
    bicudo::vec4_t<float> rect {};
    bicudo::vec2_t<float> delta {};

    float angle {};
    float angle_velocity {};
    float angle_acceleration {};

    uint64_t flags {};
  public:
    bicudo_as_signed(bicudo::body_t);
  };
}

#endif
