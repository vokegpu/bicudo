#ifndef MEOW_HPP
#define MEOW_HPP

#include "tools/pickup.hpp"
#include "graphics/immediate.hpp"

namespace meow {
  struct settings {
  public:
    bool show_aabb {};
    bool show_collision_info {};
    bool show_vertices {};
  };

  extern struct application {
  public:
    meow::immediate_graphics immediate {};
    meow::tools::pickup_info_t object_pickup_info {};
    meow::tools::pickup_info_t camera_pickup_info {};
    meow::settings settings {};
    uint64_t rendering_placements_count {};
  } app;

  void init();
  void render();
}

#endif