#ifndef BICUDO_HPP
#define BICUDO_HPP

#include "bicudo/util/log.hpp"
#include "bicudo/api/base.hpp"
#include "bicudo/physics/collision_info.hpp"
#include "bicudo/physics/placement.hpp"
#include <cstdint>

#define bicudo_version "1.3.1"

namespace bicudo {
  typedef void(*p_on_collision_pre_apply_forces)(bicudo::physics::placement*&, bicudo::physics::placement*&);
  typedef void(*p_on_collision)(bicudo::physics::placement*&, bicudo::physics::placement*&);

  struct runtime {
  public: // internal
    std::vector<bicudo::physics::placement*> placement_list {};
    bicudo::physics::collision_info_t collision_info {};
    bicudo::id highest_object_id {};
  public: // config
    float solve_accurace {1.0f};
    float intertia_const {12.0f};
    bicudo::vec2 gravity {};

    bicudo::physics_runtime_type physics_runtime_type {bicudo::physics_runtime_type::CPU_SIDE};
    bicudo::api::base *p_rocm_api {};

    bicudo::p_on_collision_pre_apply_forces p_on_collision_pre_apply_forces {nullptr};
    bicudo::p_on_collision p_on_collision {nullptr};
  };

  void init(
    bicudo::runtime *p_runtime
  );

  void quit(
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

  void update_position(
    bicudo::runtime *p_runtime,
    bicudo::physics::placement *p_placement
  );

  void update_collisions(
    bicudo::runtime *p_runtime
  );
}

#endif