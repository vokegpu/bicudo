#ifndef MEOW_HPP
#define MEOW_HPP

#include "tools/pickup.hpp"
#include "graphics/immediate.hpp"

namespace meow {
  extern struct application {
  public:
    meow::immediate_graphics immediate {};
    meow::tools::pickup_info_t pickup_info {};
  } app;

  void init();
  void render();
}

#endif