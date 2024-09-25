#include "bicudo/bicudo.hpp"
#include "bicudo/physics/processor.hpp"

#include <vector>
#include <iostream>

void bicudo::init(
  bicudo::runtime *p_runtime
) {
  bicudo::log() << "Initializing Bicudo physics simulator!";

  if (p_runtime->p_rocm_api) {
    p_runtime->p_rocm_api->init();
  }
}

void bicudo::quit(
  bicudo::runtime *p_runtime
) {
  if (p_runtime->p_rocm_api) {
    p_runtime->p_rocm_api->quit();
  }
}

void bicudo::insert(
  bicudo::runtime *p_runtime,
  bicudo::physics::placement *p_placement
) {
  bicudo::physics_placement_mass(
    p_placement,
    p_placement->mass
  );

  p_placement->vertices.resize(4);
  bicudo::splash_vertices(
    p_placement->vertices.data(),
    p_placement->pos,
    p_placement->size
  );

  p_placement->edges.resize(4);
  bicudo::splash_edges_normalized(
    p_placement->edges.data(),
    p_placement->vertices.data()
  );

  p_placement->id = p_runtime->highest_object_id++;
  p_runtime->placement_list.push_back(p_placement);
}

void bicudo::erase(
  bicudo::runtime *p_runtime,
  bicudo::physics::placement *p_placement
) {
  // meow
}

void bicudo::erase(
  bicudo::runtime *p_runtime,
  bicudo::id id
) {
  // meow
}

void bicudo::update(
  bicudo::runtime *p_runtime
) {
  bicudo::vec2 center {};
  for (bicudo::physics::placement *&p_placement : p_runtime->placement_list) {
    p_placement->acc.y = (
      p_runtime->gravity.y * (!bicudo::assert_float(p_placement->mass, 0.0f))
    ); // enable it
  
    p_placement->min.x = 99999.0f;
    p_placement->min.y = 99999.0f;
    p_placement->max.x = -99999.0f;
    p_placement->max.y = -99999.0f;

    p_placement->velocity += p_placement->acc * bicudo::dt;
    p_placement->pos += p_placement->velocity;

    p_placement->angular_velocity += p_placement->angular_acc * bicudo::dt;
    p_placement->angle += p_placement->angular_velocity;

    center.x = p_placement->pos.x + (p_placement->size.x / 2);
    center.y = p_placement->pos.y + (p_placement->size.y / 2);

    for (bicudo::vec2 &vertex : p_placement->vertices) {
      vertex += p_placement->velocity;
      vertex = vertex.rotate(p_placement->angular_velocity, center);

      p_placement->min.x = std::min(p_placement->min.x, vertex.x);
      p_placement->min.y = std::min(p_placement->min.y, vertex.y);
      p_placement->max.x = std::max(p_placement->max.x, vertex.x);
      p_placement->max.y = std::max(p_placement->max.y, vertex.y);
    }

    bicudo::splash_edges_normalized(
      p_placement->edges.data(),
      p_placement->vertices.data()
    );
  }

  bicudo::physics_processor_update(
    p_runtime
  );
}