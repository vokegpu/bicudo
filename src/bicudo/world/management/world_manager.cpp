#include "world_manager.hpp"

void bicudo::world_manager::push_back_object(bicudo::object *p_obj) {
  p_obj->id = ++this->highest_object_id;
  this->loaded_object_list.push_back(p_obj);
}

void bicudo::world_manager::on_update() {
  
}

void bicudo::world_manager::on_render() {

}