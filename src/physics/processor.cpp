#include "bicudo/physics/processor.hpp"
#include "bicudo/util/log.hpp"
#include "bicudo/bicudo.hpp"

void bicudo::physics_processor_update(
  bicudo::runtime *p_runtime
) {
  bicudo::collided was_collided {};
  bicudo::vec4 a_box {};
  bicudo::vec4 b_box {};

  float num {};
  bicudo::vec2 correction {};

  bicudo::vec2 n {};
  bicudo::vec2 start {};
  bicudo::vec2 end {};
  bicudo::vec2 p {};
  float total_mass {};

  bicudo::vec2 c1 {};
  bicudo::vec2 c2 {};

  float c1_cross {};
  float c2_cross {};

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

  uint64_t placement_size {p_runtime->placement_list.size()};
  for (uint64_t it_a {}; it_a < placement_size; it_a++) {
    /* stupid */
    for (uint64_t it_b {}; it_b < placement_size; it_b++) {
      if (it_a == it_b) {
        continue;
      }
  
      bicudo::physics::placement *&p_a {p_runtime->placement_list.at(it_a)};
      bicudo::physics::placement *&p_b {p_runtime->placement_list.at(it_b)};

      a_box.x = p_a->min.x;
      a_box.y = p_a->min.y;
      a_box.z = p_a->max.x;
      a_box.w = p_a->max.y;

      b_box.x = p_b->min.x;
      b_box.y = p_b->min.y;
      b_box.z = p_b->max.x;
      b_box.w = p_b->max.y;

      was_collided = (
        bicudo::aabb_collide_with_aabb(a_box, b_box)
      );

      if (
          (bicudo::assert_float(p_a->mass, 0.0f) && bicudo::assert_float(p_b->mass, 0.0f))
          ||
          (!was_collided)
        ) {
        continue;
      }

      p_runtime->collision_info = {};

      was_collided = (
        bicudo::physics_processor_a_collide_with_b_check(
          p_a,
          p_b,
          p_runtime
        )
      );

      if (!was_collided) {
        continue;
      }

      n = p_runtime->collision_info.normal;
      total_mass = p_a->mass + p_b->mass;
      num = p_runtime->collision_info.depth / total_mass * p_runtime->solve_accurace;
      correction = n * num;

      bicudo::physics_placement_move(
        p_a,
        correction * -p_a->mass
      );
  
      bicudo::physics_placement_move(
        p_b,
        correction * p_b->mass
      );

      if (p_runtime->p_on_collision_pre_apply_forces) {
        p_runtime->p_on_collision_pre_apply_forces(p_a, p_b);
      }

      start = p_runtime->collision_info.start * (p_b->mass / total_mass);
      end = p_runtime->collision_info.end * (p_a->mass / total_mass);
      p = start + end;

      c1.x = p_a->pos.x + (p_a->size.x / 2);
      c1.y = p_a->pos.y + (p_a->size.y / 2);
      c1 = p - c1;
    
      c2.x = p_b->pos.x + (p_b->size.x / 2);
      c2.y = p_b->pos.y + (p_b->size.y / 2);
      c2 = p - c2;

      v1 = (
        p_a->velocity + bicudo::vec2(-1.0f * p_a->angular_velocity * c1.y, p_a->angular_velocity * c1.x)
      );

      v2 = (
        p_b->velocity + bicudo::vec2(-1.0f * p_b->angular_velocity * c2.y, p_b->angular_velocity * c2.x)
      );

      vdiff = v2 - v1;
      vdiff_dot = vdiff.dot(n);

      if (vdiff_dot > 0.0f) {
        continue;
      }

      restitution = std::min(p_a->restitution, p_b->restitution);
      friction = std::min(p_a->friction, p_b->friction);
    
      c1_cross = c1.cross(n);
      c2_cross = c2.cross(n);

      jn = (
        (-(1.0f + restitution) * vdiff_dot)
        /
        (total_mass + c1_cross * c1_cross * p_a->inertia + c2_cross * c2_cross * p_b->inertia)
      );

      impulse = n * jn;

      p_a->velocity -= impulse * p_a->mass;
      p_b->velocity += impulse * p_b->mass;

      p_a->angular_velocity -= c1_cross * jn * p_a->inertia;
      p_b->angular_velocity += c2_cross * jn * p_b->inertia;

      tangent = vdiff - n * vdiff_dot;
      tangent = tangent.normalize() * -1.0f;

      c1_cross = c1.cross(tangent);
      c2_cross = c2.cross(tangent);

      jt = (
        (-(1.0f + restitution) * vdiff.dot(tangent) * friction)
        /
        (total_mass + c1_cross * c1_cross * p_a->inertia + c2_cross * c2_cross * p_b->inertia)
      );

      jt = jt > jn ? jn : jt;
      impulse = tangent * jt;
  
      p_a->velocity -= impulse * p_a->mass;
      p_b->velocity += impulse * p_b->mass;
  
      p_a->angular_velocity -= c1_cross * jt * p_a->inertia;
      p_b->angular_velocity += c2_cross * jt * p_b->inertia;

      if (p_runtime->p_on_collision) {
        p_runtime->p_on_collision(p_a, p_b);
      }
    }
  }
}

bicudo::collided bicudo::physics_processor_a_collide_with_b_check(
  bicudo::physics::placement *&p_a,
  bicudo::physics::placement *&p_b,
  bicudo::runtime *p_runtime
) {
  p_runtime->collision_info.collided = false;

  switch (p_runtime->physics_runtime_type) {
    case bicudo::physics_runtime_type::CPU_SIDE: {
      bicudo::physics::collision_info_t a_collision_info {};
      bicudo::physics_processor_find_axis_penetration(
        p_a,
        p_b,
        &a_collision_info
      );

      if (!a_collision_info.has_support_point) {
        return p_runtime->collision_info.collided;
      }

      bicudo::physics::collision_info_t b_collision_info {};
      bicudo::physics_processor_find_axis_penetration(
        p_b,
        p_a,
        &b_collision_info
      );
    
      if (!b_collision_info.has_support_point) {
        return p_runtime->collision_info.collided;
      }
    
      if (a_collision_info.depth < b_collision_info.depth) {
        p_runtime->collision_info.depth = a_collision_info.depth;
        p_runtime->collision_info.normal = a_collision_info.normal;
        p_runtime->collision_info.start = (
          a_collision_info.start - (a_collision_info.normal * a_collision_info.depth)
        );
    
        bicudo::physics_processor_collision_info_update(
          &p_runtime->collision_info
        );
      } else {
        p_runtime->collision_info.depth = b_collision_info.depth;
        p_runtime->collision_info.normal = b_collision_info.normal * -1.0f;
        p_runtime->collision_info.start = b_collision_info.start;
    
        bicudo::physics_processor_collision_info_update(
          &p_runtime->collision_info
        );
      }

      return (
        p_runtime->collision_info.collided = true
      );
    }

    case bicudo::physics_runtime_type::GPU_ROCM: {
      p_runtime->p_rocm_api->update_physics_simulator(
        p_a,
        p_b,
        &p_runtime->collision_info
      );

      return p_runtime->collision_info.collided;
    }
  }

  return p_runtime->collision_info.collided;
}

void bicudo::physics_processor_collision_info_change_dir(
  bicudo::physics::collision_info_t *p_collision_info
) {
  p_collision_info->normal *= -1.0f;
  bicudo::vec2 n {p_collision_info->normal};
  p_collision_info->start = p_collision_info->end;
  p_collision_info->end = n;
}

void bicudo::physics_processor_collision_info_update(
  bicudo::physics::collision_info_t *p_collsion_info
) {
  p_collsion_info->end = (
    p_collsion_info->start + p_collsion_info->normal * p_collsion_info->depth
  );
}

void bicudo::physics_processor_find_axis_penetration(
  bicudo::physics::placement *&p_a,
  bicudo::physics::placement *&p_b,
  bicudo::physics::collision_info_t *p_collision_info
) {
  bicudo::vec2 edge {};
  bicudo::vec2 support_point {};

  float best_dist {FLT_MAX};
  uint64_t best_edge {};
  p_collision_info->has_support_point = true;

  bicudo::vec2 dir {};
  bicudo::vec2 vert {};
  bicudo::vec2 to_edge {};

  float proj {};
  float dist {};

  bicudo::vec2 point {};

  uint64_t edges_size {p_a->edges.size()};
  uint64_t vertices_size {p_b->vertices.size()};

  for (uint64_t it_edges {}; p_collision_info->has_support_point && it_edges < edges_size; it_edges++) {
    edge = p_a->edges.at(it_edges); // normalized edge
    dir = edge * -1.0f;
    vert = p_a->vertices.at(it_edges);

    dist = -FLT_MAX;
    p_collision_info->has_support_point = false;

    for (bicudo::vec2 &vertex : p_b->vertices) {
      to_edge = vertex - vert;
      proj = to_edge.dot(dir);

      if (proj > 0.0f && proj > dist) {
        point = vertex;
        dist = proj;
        p_collision_info->has_support_point = true;
      }
    }

    if (p_collision_info->has_support_point && dist < best_dist) {
      best_dist = dist;
      best_edge = it_edges;
      support_point = point;
    }
  }

  if (p_collision_info->has_support_point) {
    edge = p_a->edges.at(best_edge);

    p_collision_info->depth = best_dist;
    p_collision_info->normal = edge;
    p_collision_info->start = support_point + (edge * best_dist);

    bicudo::physics_processor_collision_info_update(
      p_collision_info
    );
  }
}