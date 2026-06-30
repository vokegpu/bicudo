#ifndef MEOW_WORLD_CAMERA_HPP
#define MEOW_WORLD_CAMERA_HPP

#include <bicudo/physics/rect.hpp>

namespace meow {
  class camera {
  public:
    bicudo::vec4_t<float> view {};
    bicudo::rect_t<float> rect {};
    float smooth_amount {0.2f};
    float zoom {1.0f};
    float interpolated_zoom {1.0f};
  public:
    void create();
    void on_update();
  };
}

#endif
