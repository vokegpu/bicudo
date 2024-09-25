#ifndef BICUDO_HPP
#define BICUDO_HPP

#include "bicudo/util/log.hpp"
#include "bicudo/api/base.hpp"
#include "bicudo/physics/collision_info.hpp"
#include "bicudo/physics/placement.hpp"
#include <cstdint>

namespace bicudo {
  struct runtime {
  public:
    std::vector<bicudo::physics::placement*> placement_list {};
    bicudo::physics::collision_info_t collision_info {};
    bicudo::id highest_object_id {};
  public:
    bicudo::vec2 gravity {};
    bicudo::physics_runtime_type physics_runtime_type {bicudo::physics_runtime_type::CPU_SIDE};
    bicudo::api::base *p_rocm_api {};
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
}

#endif