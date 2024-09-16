#ifndef BICUDO_WORLD_CAMERA_HPP
#define BICUDO_WORLD_CAMERA_HPP

#include "bicudo/util/math.hpp"

namespace bicudo {
  class camera {
  public:
    static bicudo::vec2 display;
  public:
    bicudo::placement placement {};
    float smooth_amount {0.2f};
  public:
    void create();
    void set_viewport(int32_t w, int32_t h);
    void on_update();
  };
}

#endif