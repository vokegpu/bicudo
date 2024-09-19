#include "bicudo/world/management/world_manager.hpp"
#include "bicudo/util/log.hpp"

void bicudo::world_manager::push_back_object(bicudo::object *p_obj) {
  p_obj->id = ++this->highest_object_id;
  this->loaded_object_list.push_back(p_obj);
  this->simulator.placement_list.push_back(&p_obj->placement);
}

void bicudo::world_manager::on_create() {
  bicudo::world_physics_init(
    &this->simulator
  );
}

void bicudo::world_manager::on_update() {
  for (bicudo::object *&p_objs : this->loaded_object_list) {
    p_objs->placement.acc.y = (gravity.y * (!bicudo::assert_float(p_objs->placement.mass, 0.0f))); // enable it
    p_objs->on_update();
  }

  bicudo::world_physics_update_simulator(
    &this->simulator
  );

  this->camera.on_update();
}