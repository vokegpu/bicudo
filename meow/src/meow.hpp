#ifndef MEOW_HPP
#define MEOW_HPP

#include "tools/pickup.hpp"
#include "graphics/immediate.hpp"

namespace meow {
  extern struct application {
  public:
    meow::immediate_graphics immediate {};
    meow::tools::pickup_info_t object_pickup_info {};
    meow::tools::pickup_info_t camera_pickup_info {};
    uint64_t rendering_placements_count {};
  } app;

  void init();
  void render();
}

#endif