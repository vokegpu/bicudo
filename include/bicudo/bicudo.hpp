#ifndef BICUDO_HPP
#define BICUDO_HPP

#include "bicudo/util/log.hpp"
#include "bicudo/world/management/world_manager.hpp"
#include "bicudo/gpu/model.hpp"
#include <cstdint>

namespace bicudo {
  extern struct application {
  public:
    bicudo::world_manager world_manager {};
  } app;

  void init();
  void update();
  void viewport(int32_t w, int32_t h);

  namespace world {
    bicudo::camera &camera();
    void insert(bicudo::object *p_obj);
    bicudo::collided pick(bicudo::object *&p_obj, bicudo::vec2 pos);
  }

  void count(uint32_t *p_number);
}

#endif