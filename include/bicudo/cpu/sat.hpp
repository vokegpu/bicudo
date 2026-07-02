#ifndef BICUDO_CPU_SAT_CPU_HPP
#define BICUDO_CPU_SAT_CPU_HPP

#include <bicudo/physics/hypergroup.hpp>

namespace bicudo {
  struct cpu_sat_collide_info_t {
  public:
    bicudo::vec2_t<float> dir {};
    bicudo::vec2_t<float> start {};
    bicudo::vec2_t<float> end {};
    float depth {};
    bool collided {};
  public:
    operator bool() {
      return this->collided;
    }
  };
}

namespace bicudo {
  bicudo::cpu_sat_collide_info_t cpu_sat_check_collide(
    bicudo::body_t &a,
    bicudo::body_t &b
  );

  bicudo::cpu_sat_collide_info_t cpu_sat_collided(
    bicudo::body_t &a,
    bicudo::body_t &b
  );

  void cpu_sat_update_body(
    bicudo::body_t &body
  );
}

#endif
