#include "bicudo/bicudo.hpp"
#include "bicudo/physics/processor.hpp"

#include <vector>
#include <iostream>

void bicudo::init(
  bicudo::runtime *p_runtime
) {
  bicudo::log() << "Initializing Bicudo " << bicudo_version << " physics simulator!";

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
  bicudo::physics::placement *p_placement,
  bool free
) {
  for (uint64_t it {}; it < p_runtime->placement_list.size(); it++) {
    bicudo::physics::placement *p_alive_placement {
      p_runtime->placement_list.at(it)
    };

    if (p_alive_placement != nullptr && p_alive_placement == p_placement) {
      if (free) {
        delete p_alive_placement;
        p_alive_placement = nullptr;
      }

      p_runtime->placement_list.erase(
        p_runtime->placement_list.begin() + it
      );

      break;
    }
  }
}

void bicudo::erase(
  bicudo::runtime *p_runtime,
  bicudo::id id,
  bool free
) {
  for (uint64_t it {}; it < p_runtime->placement_list.size(); it++) {
    bicudo::physics::placement *p_alive_placement {
      p_runtime->placement_list.at(it)
    };

    if (p_alive_placement != nullptr && p_alive_placement->id == id) {
      if (free) {
        delete p_alive_placement;
        p_alive_placement = nullptr;
      }

      p_runtime->placement_list.erase(
        p_runtime->placement_list.begin() + it
      );

      break;
    }
  }
}

void bicudo::update(
  bicudo::runtime *p_runtime
) {
  for (bicudo::physics::placement *&p_placement : p_runtime->placement_list) {
    bicudo::update_position(
      p_runtime, 
      p_placement
    );
  }

  bicudo::physics_processor_update(
    p_runtime
  );
}

void bicudo::update_position(
  bicudo::runtime *p_runtime,
  bicudo::physics::placement *p_placement
) {
  bicudo::vec2 center {};
  if ( // temp
      !bicudo::assert_float(p_placement->prev_size.x, p_placement->size.x)
      ||
      !bicudo::assert_float(p_placement->prev_size.y, p_placement->size.y)
    ) {
    p_placement->prev_size = p_placement->size;
    bicudo::splash_vertices(
      p_placement->vertices.data(),
      p_placement->pos,
      p_placement->size
    );
  }

  if ( // temp
      !bicudo::assert_float(p_placement->prev_mass, p_placement->mass)
    ) {
    if (bicudo::assert_float(p_placement->mass, 0.0f)) {
      p_placement->inertia = 0.0f;
      p_placement->mass = 0.0f;
    } else {
      p_placement->mass = (1.0f/ p_placement->mass);
      p_placement->inertia = (1.0f / p_placement->mass) * p_placement->size.magnitude_no_sq() / p_runtime->intertia_const;
      p_placement->inertia = 1.0f / p_placement->inertia;
    }

    p_placement->prev_mass = p_placement->mass;
  }

  p_placement->acc.y = (
    (p_runtime->gravity.y)
    *
    (!bicudo::assert_float(p_placement->mass, 0.0f))
    *
    (!p_placement->turn_off_gravity)
  ); // enable it

  p_placement->min.x = 99999.0f;
  p_placement->min.y = 99999.0f;
  p_placement->max.x = -99999.0f;
  p_placement->max.y = -99999.0f;

  p_placement->velocity += p_placement->acc * bicudo::dt;
  p_placement->pos += p_placement->velocity;

  p_placement->angular_velocity += p_placement->angular_acc * bicudo::dt;
  p_placement->angle += p_placement->angular_velocity;

  center = p_placement->pos + (p_placement->size / 2.0f);
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

void bicudo::update_collisions(
  bicudo::runtime *p_runtime
) {
  bicudo::physics_processor_update(
    p_runtime
  );
}