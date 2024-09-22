#ifndef MEOW_TOOLS_CAMERA_HPP
#define MEOW_TOOLS_CAMERA_HPP

#include "bicudo/util/math.hpp"
#include "bicudo/physics/placement.hpp"

namespace meow {
  class camera {
  public:
    bicudo::physics::placement placement {};
    float smooth_amount {0.2f};
    bicudo::vec4 rect {};
    float zoom {1.0f};
    float interpolated_zoom {1.0f};
  public:
    void create();
    void on_update();
  };
}

#endif