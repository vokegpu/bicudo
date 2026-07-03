#ifndef MEOW_WORLD_CAMERA_HPP
#define MEOW_WORLD_CAMERA_HPP

#include <bicudo/physics/body.hpp>

namespace meow {
  class camera {
  public:
    bicudo::vec4_t<float> view {};
    bicudo::body_t rect {};
    float smooth_amount {0.2f};
    float zoom {1.0f};
    float interpolated_zoom {1.0f};
    bool is_while_zoom {};
  public:
    void set_zoom(float zoom);
    void create();
    void on_update();
  };
}

#endif
