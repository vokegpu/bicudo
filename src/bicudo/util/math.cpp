#include "math.hpp"

uint64_t bicudo::framerate {75};
uint64_t bicudo::current_framerate {1};
uint64_t bicudo::cpu_interval_ms {};
float bicudo::dt {0.016f};

void bicudo::set_framerate(uint64_t wish_fps) {
  bicudo::framerate = wish_fps;
  bicudo::cpu_interval_ms = 1000 / bicudo_clamp(bicudo::framerate, 1, 999);
}

void bicudo::splash_vertices(
  bicudo::vec2 *p_vertices,
  bicudo::vec2 &pos,
  bicudo::vec2 &size
) {
  float w {size.x};
  float h {size.y};

  p_vertices[0] = pos;
  p_vertices[1] = bicudo::vec2 {pos.x + w, pos.y};
  p_vertices[2] = bicudo::vec2 {pos.x + w, pos.y + h};
  p_vertices[3] = bicudo::vec2 {pos.x, pos.y + h};
}

void bicudo::splash_edges_normalized(
  bicudo::vec2 *p_edges,
  bicudo::vec2 *p_vertices
) {
  p_edges[0] = (p_vertices[1] - p_vertices[2]).normalize();
  p_edges[1] = (p_vertices[2] - p_vertices[3]).normalize();
  p_edges[2] = (p_vertices[3] - p_vertices[0]).normalize();
  p_edges[3] = (p_vertices[0] - p_vertices[1]).normalize();
}

bicudo::mat4 bicudo::ortho(float left, float right, float bottom, float top) {
  float far {1.0f};
  float near {-1.0};

  return bicudo::mat4 {
    2.0f / (right - left),        0.0f,                         0.0f,                     0.0f,
    0.0f,                         2.0f / (top - bottom),        0.0f,                     0.0f,
    0.0f,                         0.0f,                         (-2.0f)/(far - near),     0.0f,
    -((right+left)/(right-left)), -((top+bottom)/(top-bottom)), -((far+near)/(far-near)), 1.0f
  };
}

bool bicudo::vec4_collide_with_vec2(const bicudo::vec4 &vec4, const bicudo::vec2 &vec2) {
  return (
    (vec2.x > vec4.x && vec2.x < vec4.x + vec4.z)
    &&
    (vec2.y > vec4.y && vec2.y < vec4.y + vec4.w)
  );
}