#ifndef BICUDO_HPP
#define BICUDO_HPP

#include "bicudo/util/log.hpp"
#include "bicudo/gpu/model.hpp"
#include "bicudo/physics/simulator.hpp"
#include "bicudo/physics/placement.hpp"
#include <cstdint>

namespace bicudo {
  struct runtime {
  public:
    bicudo::physics::simulator simulator {};
    bicudo::id highest_object_id {};
    bicudo::vec2 gravity {};
    bicudo::physics_runtime_type physics_runtime_type {}; 
  };

  void init(
    bicudo::runtime *p_runtime
  );

  void insert(
    bicudo::runtime *p_runtime,
    bicudo::physics::placement *p_placement
  );

  void erase(
    bicudo::runtime *p_runtime,
    bicudo::physics::placement *p_placement
  );

  void erase(
    bicudo::runtime *p_runtime,
    bicudo::id id
  );

  void update(
    bicudo::runtime *p_runtime
  );
}

#endif