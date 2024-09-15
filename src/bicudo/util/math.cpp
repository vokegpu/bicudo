#include "math.hpp"
#include <iostream>

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

/**
 * From:
 * https://en.wikipedia.org/wiki/Orthographic_projection
 **/
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

/**
 * From:
 * https://gist.github.com/yiwenl/3f804e80d0930e34a0b33359259b556c
 **/
bicudo::mat4 bicudo::rotate(bicudo::mat4 mat, bicudo::vec3 axis, float angle) {
  axis = axis.normalize();

  float s {std::sin(angle)};
  float c {std::cos(angle)};
  float oc {1.0f - c};

  bicudo::mat4 rotate = bicudo::mat4 {
    oc * axis.x * axis.x + c,          oc * axis.x * axis.y - axis.z * s, oc * axis.z * axis.x + axis.y * s, 0.0f,
    oc * axis.x * axis.y + axis.z * s, oc * axis.y * axis.y + c,          oc * axis.y * axis.z - axis.x * s, 0.0f,
    oc * axis.z * axis.x - axis.y * s, oc * axis.y * axis.z + axis.x * s, oc * axis.z * axis.z + c,          0.0f,
    0.0f,                              0.0f,                              0.0f,                              1.0f
  };

  return mat * rotate;
}

bicudo::mat4 bicudo::translate(bicudo::mat4 mat, bicudo::vec2 pos) {
  bicudo::mat4 translate {1.0f};
  translate[12] = pos.x;
  translate[13] = pos.y;
  translate[14] = 0.0f;
  return mat * translate;
}

bool bicudo::aabb_collide_with_vec2(const bicudo::vec2 &min, const bicudo::vec2 &max, const bicudo::vec2 &vec2) {
  return (
    vec2.x > min.x && vec2.y > min.y && vec2.x < max.x && vec2.y < max.y
  );
}

bool bicudo::vec4_collide_with_vec2(const bicudo::vec4 &vec4, const bicudo::vec2 &vec2) {
  return (
    (vec2.x > vec4.x && vec2.x < vec4.x + vec4.z)
    &&
    (vec2.y > vec4.y && vec2.y < vec4.y + vec4.w)
  );
}

void bicudo::move(bicudo::placement *p_placement, const bicudo::vec2 &dir) {
  p_placement->min.x = 99999.0f;
  p_placement->min.y = 99999.0f;
  p_placement->max.x = -99999.0f;
  p_placement->max.y = -99999.0f;

  for (bicudo::vec2 &vertex : p_placement->vertices) {
    vertex += dir;

    p_placement->min.x = std::min(p_placement->min.x, vertex.x);
    p_placement->min.y = std::min(p_placement->min.y, vertex.y);
    p_placement->max.x = std::max(p_placement->max.x, vertex.x);
    p_placement->max.y = std::max(p_placement->max.y, vertex.y);
  }

  bicudo::splash_edges_normalized(
    p_placement->edges.data(),
    p_placement->vertices.data()
  );

  p_placement->pos += dir;
}

void bicudo::rotate(bicudo::placement *p_placement, float angle_dir) {
  bicudo::vec2 center {
    p_placement->pos.x + (p_placement->size.x / 2),
    p_placement->pos.y + (p_placement->size.y / 2)
  };

  for (bicudo::vec2 &vertex : p_placement->vertices) {
    vertex = vertex.rotate(angle_dir, center);
  }

  p_placement->angle += angle_dir;
}

void bicudo::mass(bicudo::placement *p_placement, float mass) {
  if (bicudo::assert_float(mass, 0.0f)) {
    p_placement->inertia = 0.0f;
    p_placement->mass = 0.0f;
  } else {
    p_placement->mass = mass;
    p_placement->inertia = (1.0f / mass) * p_placement->size.magnitude_no_sq() / 12;
    p_placement->inertia = 1.0f / p_placement->inertia;
  }
}