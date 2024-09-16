#ifndef BICUDO_WORLD_MANAGEMENT_WORLD_MANAGER_HPP
#define BICUDO_WORLD_MANAGEMENT_WORLD_MANAGER_HPP

#include "bicudo/world/types.hpp"
#include "bicudo/world/object.hpp"
#include "bicudo/world/camera.hpp"
#include "bicudo/world/physics/simulator.hpp"

#include <vector>

namespace bicudo {
  class world_manager {
  public:
    bicudo::id highest_object_id {};
    bicudo::world::physics::simulator simulator {};
    bicudo::vec2 gravity {};
    bicudo::camera camera {};
  public:
    std::vector<bicudo::object*> loaded_object_list {};
  public:
    void push_back_object(bicudo::object *p_obj);
    void on_create();
    void on_update();
  };
}

#endif
