#include "world_manager.hpp"

void bicudo::world_manager::push_back_object(bicudo::object *p_obj) {
  p_obj->id = ++this->highest_object_id;
  this->loaded_object_list.push_back(p_obj);
}

void bicudo::world_manager::on_event(SDL_Event &sdl_event) {
  if (sdl_event.type == SDL_WINDOWEVENT && sdl_event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
    this->immediate.set_viewport(sdl_event.window.data1, sdl_event.window.data2);
  }
}

void bicudo::world_manager::on_create() {
  this->immediate.create();
}

void bicudo::world_manager::on_update() {
  bicudo::world_physics_update_simulator(
    &this->simulator
  );
}

void bicudo::world_manager::on_render() {
  this->immediate.invoke();
  this->immediate.draw({20, 20, 400, 400}, {0.5f, 0.5f, 0.5f, 1.0f});
  this->immediate.revoke();
}