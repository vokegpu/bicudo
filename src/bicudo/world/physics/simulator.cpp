#include "simulator.hpp"
#include <iostream>

void bicudo::world_physics_update_simulator(
  bicudo::world::physics::simulator *p_simulator
) {
  bicudo::world::physics::support_info_t support_info {};
  bicudo::collided was_collided {};

  float num {};
  bicudo::vec2 correction {};

  bicudo::vec2 n {};
  bicudo::vec2 start {};
  bicudo::vec2 end {};
  bicudo::vec2 p {};
  float total_mass {};

  bicudo::vec2 c1 {};
  bicudo::vec2 c2 {};

  bicudo::vec2 v1 {};
  bicudo::vec2 v2 {};
  bicudo::vec2 vdiff {};
  float vdiff_dot {};

  float restitution {};
  float friction {};
  float jn {};
  float jt {};

  bicudo::vec2 impulse {};
  bicudo::vec2 tangent {};

  uint64_t placement_size {p_simulator->placement_list.size()};
  for (uint64_t it_a {}; it_a < placement_size; it_a++) {
    /* stupid */
    for (uint64_t it_b {}; it_b < placement_size; it_b++) {
      if (it_a == it_b) {
        continue;
      }

      bicudo::placement *&p_a {p_simulator->placement_list.at(it_a)};
      bicudo::placement *&p_b {p_simulator->placement_list.at(it_b)};

      if (bicudo::assert_float(p_a->mass, 0.0f) && bicudo::assert_float(p_b->mass, 0.0f)) {
        continue;
      }

      was_collided = (
        bicudo::world_physics_a_collide_with_b_check(
          p_a,
          p_b,
          &p_simulator->collision_info,
          &support_info
        )
      );

      p_a->was_collided = was_collided;
      p_b->was_collided = was_collided;

      if (!was_collided) {
        continue;
      }

      num = p_simulator->collision_info.depth / (p_a->mass + p_b->mass) * 0.8f;
      correction = p_simulator->collision_info.normal * num;

      bicudo::move(
        p_a,
        correction * -p_a->mass
      );

      bicudo::move(
        p_b,
        correction * p_b->mass
      );

      
      /*
       angular x (rotate-matrix not done yet)

      n = p_simulator->collision_info.normal;
      start = p_simulator->collision_info.start * (p_b->masss / (p_a->mass + p_b->mass));
      end = p_simulator->collision_info.end * (p_a->masss / (p_a->mass + p_b->mass));
      p = start + end;

      c1.x = p_a->pos.x + (p_a->size.x / 2);
      c1.y = p_a->pos.y + (p_a->size.y / 2);
      c1 = p - c1;
    
      c2.x = p_b->pos.x + (p_b->size.x / 2);
      c2.y = p_b->pos.y + (p_b->size.y / 2);
      c2 = p - c2;

      v1 = p_a->velocity + {-1.0f * p_a->angular_velocity};
      */

      total_mass = p_a->mass + p_b->mass;
      n = p_simulator->collision_info.normal;
      v1 = p_a->velocity;
      v2 = p_b->velocity;

      vdiff = v2 - v1;
      vdiff_dot = vdiff.dot(n);

      if (vdiff_dot > 0.0f) {
        continue;
      }

      restitution = p_a->restitution < p_b->restitution ? p_a->restitution : p_b->restitution;
      friction = p_a->restitution < p_b->restitution ? p_a->restitution : p_b->restitution;

      jn = (
        (-(1.0f + restitution) * vdiff_dot) / (total_mass)
      );

      impulse = n * jn;

      p_a->velocity -= impulse * p_a->mass;          
      p_b->velocity += impulse * p_b->mass;

      tangent = vdiff - n * vdiff_dot;
      tangent = tangent.normalize() * -1.0f;

      jt = (
        (-(1.0f + restitution) * vdiff.dot(tangent) * friction)
        /
        (total_mass)
      );

      jt = jt > jn ? jn : jt;
      impulse = tangent * jt;

      p_a->velocity -= impulse * p_a->mass;
      p_b->velocity += impulse * p_b->mass;
    }
  }
}

bicudo::collided bicudo::world_physics_a_collide_with_b_check(
  bicudo::placement *&p_a,
  bicudo::placement *&p_b,
  bicudo::world::physics::collision_info_t *p_collision_info,
  bicudo::world::physics::support_info_t *p_support_info
) {
  bicudo::collided a_has_support_point {};
  bicudo::world::physics::collision_info_t a_collision_info {};

  bicudo::world_physics_find_axis_penetration(
    p_a,
    p_b,
    &a_collision_info,
    p_support_info,
    &a_has_support_point
  );

  if (!a_has_support_point) {
    return false;
  }

  bicudo::collided b_has_support_point {};
  bicudo::world::physics::collision_info_t b_collision_info {};

  bicudo::world_physics_find_axis_penetration(
    p_b,
    p_a,
    &b_collision_info,
    p_support_info,
    &b_has_support_point
  );

  if (a_collision_info.depth < b_collision_info.depth) {
    p_collision_info->depth = a_collision_info.depth;
    p_collision_info->normal = a_collision_info.normal;
    p_collision_info->start = a_collision_info.start - (a_collision_info.normal * a_collision_info.depth);

    bicudo::world_physics_collision_info_update(
      p_collision_info
    );
  } else {
    p_collision_info->depth = b_collision_info.depth;
    p_collision_info->normal = b_collision_info.normal * -1.0f;
    p_collision_info->start = b_collision_info.start;

    bicudo::world_physics_collision_info_update(
      p_collision_info
    );
  }

  return true;
}

void bicudo::world_physics_collision_info_change_dir(
  bicudo::world::physics::collision_info_t *p_collision_info
) {
  p_collision_info->normal *= -1.0f;
  bicudo::vec2 n {p_collision_info->normal};
  p_collision_info->start = p_collision_info->end;
  p_collision_info->end = n;
}

void bicudo::world_physics_collision_info_update(
  bicudo::world::physics::collision_info_t *p_collsion_info
) {
  p_collsion_info->end = (
    p_collsion_info->start + p_collsion_info->normal * p_collsion_info->depth
  );
}

void bicudo::world_physics_find_axis_penetration(
  bicudo::placement *&p_a,
  bicudo::placement *&p_b,
  bicudo::world::physics::collision_info_t *p_collision_info,
  bicudo::world::physics::support_info_t *p_support_info,
  bool *p_has_support_point
) {
  bicudo::vec2 edge {};
  bicudo::vec2 vertex {};
  bicudo::vec2 support_point {};

  float best_dist {99999.0f};
  uint64_t best_edge {};
  *p_has_support_point = true;

  bicudo::vec2 dir {};
  bicudo::vec2 vert {};
  bicudo::vec2 to_edge {};

  float proj {};

  uint64_t edges_size {p_a->edges.size()};
  uint64_t vertices_size {p_b->vertices.size()};

  for (uint64_t it_edges {}; *p_has_support_point && it_edges < edges_size; it_edges++) {
    edge = p_a->edges.at(it_edges); // normalized edge
    dir = edge * -1.0f;
    vert = p_a->vertices.at(it_edges);

    p_support_info->distance = -99999.0f;
    p_support_info->point = {};
    *p_has_support_point = false;

    for (uint64_t it_vertex {}; it_vertex < vertices_size; it_vertex++) {
      vertex = p_b->vertices.at(it_vertex);
      to_edge = vertex - vert;
      proj = to_edge.dot(dir);

      if (proj > 0.0f && proj > p_support_info->distance) {
        p_support_info->point = vertex;
        p_support_info->distance = proj;
        *p_has_support_point = true;
      }
    }

    if (*p_has_support_point && p_support_info->distance < best_dist) {
      best_dist = p_support_info->distance;
      best_edge = it_edges;
      support_point = p_support_info->point;
    }
  }

  if (*p_has_support_point) {
    edge = p_a->edges.at(best_edge);

    p_collision_info->depth = best_dist;
    p_collision_info->normal = edge;
    p_collision_info->start = p_support_info->point + (edge * best_dist);

    bicudo::world_physics_collision_info_update(
      p_collision_info
    );
  }
}