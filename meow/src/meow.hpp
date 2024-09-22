#ifndef MEOW_HPP
#define MEOW_HPP

#include "tools/pickup.hpp"
#include "tools/camera.hpp"
#include "bicudo/bicudo.hpp"
#include "graphics/immediate.hpp"

#define MEOW_INITIAL_WINDOW_OFFSET 89

namespace meow {
  struct settings {
  public:
    bool show_aabb {};
    bool show_collision_info {};
    bool show_vertices {};
  };

  extern struct application {
  public:
    bicudo::runtime bicudo {};
    meow::camera camera {};
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