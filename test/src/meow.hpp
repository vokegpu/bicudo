#ifndef MEOW_HPP
#define MEOW_HPP

#include "world/camera.hpp"
#include "graphics/graphics.hpp"

#include <bicudo/bicudo.hpp>

namespace meow {
  extern struct application_t {
  public:
    bicudo::core_t bicudo {};
  public:
    meow::camera camera {};
    meow::immediate_graphics immediate {};
  } app;
}

#endif
