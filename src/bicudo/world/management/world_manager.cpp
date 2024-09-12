#include "world_manager.hpp"
#include "bicudo/util/log.hpp"

void bicudo::world_manager::push_back_object(bicudo::object *p_obj) {
  p_obj->id = ++this->highest_object_id;
  this->loaded_object_list.push_back(p_obj);
  this->simulator.placement_list.push_back(&p_obj->placement);
}

void bicudo::world_manager::on_create() {
  this->immediate.create();
}

void bicudo::world_manager::on_update() {
  for (bicudo::object *&p_objs : this->loaded_object_list) {
    p_objs->on_update();
  }

  bicudo::world_physics_update_simulator(
    &this->simulator
  );
}

void bicudo::world_manager::on_render() {
  this->immediate.invoke();

  bicudo::vec4 rect {};
  bicudo::vec4 color(0.3f, 0.5f, 0.675f, 1.0f);

  for (bicudo::object *&p_objs : this->loaded_object_list) {
    rect.x = p_objs->placement.pos.x;
    rect.y = p_objs->placement.pos.y;

    rect.z = p_objs->placement.size.x;
    rect.w = p_objs->placement.size.y;

    color.x = p_objs->placement.was_collided;
    this->immediate.draw(rect, color);
  }

  this->immediate.revoke();
}