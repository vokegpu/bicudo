#include "math.hpp"

uint64_t bicudo::framerate {75};
uint64_t bicudo::current_framerate {1};
uint64_t bicudo::cpu_interval_ms {};
float bicudo::dt {0.016f};

void bicudo::set_framerate(uint64_t wish_fps) {
  bicudo::framerate = wish_fps;
  bicudo::cpu_interval_ms = 1000 / bicudo_clamp(bicudo::framerate, 1, 999);
}

bicudo::mat4 bicudo::ortho(float left, float top, float bottom, float right) {
  float far {1.0f};
  float near {-1.0f};

  return bicudo::mat4 {
    2.0f / (right - left), 0.0f,                  0.0f,                 -((right+left)/(right-left)),
    0.0f,                  2.0f / (top - bottom), 0.0f,                 -((top+bottom)/(top-bottom)),
    0.0f,                  0.0f,                  (-2.0f)/(far - near), -((far+near)/(far-near)),
    0.0f,                  0.0f,                  0.0f,                  1.0f
  };
}