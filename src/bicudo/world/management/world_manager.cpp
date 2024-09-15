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
    p_objs->placement.acc.y = (gravity.y * (!bicudo::assert_float(p_objs->placement.mass, 0.0f))); // enable it
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
    this->immediate.draw(rect, color, p_objs->placement.angle);

    this->immediate.draw({p_objs->placement.min.x, p_objs->placement.min.y, 10.0f, 10.0f}, {0.0f, 1.0f, 1.0f, 1.0f}, 0.0f);
    this->immediate.draw({p_objs->placement.max.x, p_objs->placement.max.y, 10.0f, 10.0f}, {0.0f, 0.0f, 1.0f, 1.0f}, 0.0f);

    this->immediate.draw({p_objs->placement.vertices[0].x, p_objs->placement.vertices[0].y, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
    this->immediate.draw({p_objs->placement.vertices[1].x, p_objs->placement.vertices[1].y, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
    this->immediate.draw({p_objs->placement.vertices[2].x, p_objs->placement.vertices[2].y, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
    this->immediate.draw({p_objs->placement.vertices[3].x, p_objs->placement.vertices[3].y, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
    this->immediate.draw({rect.x + p_objs->placement.size.x / 2, rect.y + p_objs->placement.size.y / 2, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
  }

  rect.z = 10.0f;
  rect.w = 10.0f;

  rect.x = this->simulator.collision_info.start.x;
  rect.y = this->simulator.collision_info.start.y;

  this->immediate.draw(rect, {1.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

  rect.x = this->simulator.collision_info.end.x;
  rect.y = this->simulator.collision_info.end.y;

  this->immediate.draw(rect, {0.0f, 1.0f, 0.0f, 1.0f}, 0.0f);
  this->immediate.revoke();
}