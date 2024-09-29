#include "bicudo/physics/placement.hpp"

void bicudo::physics_placement_move(
  bicudo::physics::placement *p_placement,
  const bicudo::vec2 &dir
) {
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

void bicudo::physics_placement_rotate(
  bicudo::physics::placement *p_placement,
  float angle_dir
) {
  bicudo::vec2 center {
    p_placement->pos.x + (p_placement->size.x / 2),
    p_placement->pos.y + (p_placement->size.y / 2)
  };

  for (bicudo::vec2 &vertex : p_placement->vertices) {
    vertex = vertex.rotate(angle_dir, center);
  }

  p_placement->angle += angle_dir;
}