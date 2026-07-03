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

    float angle {};
    float angular_velocity {};
    float angle_acceleration {};

    float inertia {0.0f};
    float mass {1.0f};
    float friction {0.8f};
    float restitution {0.2f};

    uint64_t flags {};
    bool has_collide {};
    bool no_gravity {false};

    std::vector<bicudo::vec2_t<float>> edges {};
    std::vector<bicudo::vec2_t<float>> vertices {};

    bicudo::vec4_t<float> rect {};
    bicudo::vec2_t<float> delta {};
    bicudo::vec2_t<float> min {};
    bicudo::vec2_t<float> max {};
  public:
    bicudo_as_signed(bicudo::body_t);
  };
}

namespace bicudo {
  void size(bicudo::vec2_t<float> size);
  void move(bicudo::vec2_t<float> direction);
}

#endif
