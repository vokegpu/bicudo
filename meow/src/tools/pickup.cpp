#include "pickup.hpp"

#include "bicudo/bicudo.hpp"
#include <ekg/ekg.hpp>
#include "meow.hpp"

void meow::tools_to_local_camera(bicudo::vec2 *p_vec) {
  p_vec->x /= meow::app.camera.zoom;
  p_vec->y /= meow::app.camera.zoom;
}

bicudo::collided meow::tools_pick_physics_placement(bicudo::physics::placement *&p_placement, bicudo::vec2 pos) {
  bicudo::vec2 &cam {meow::app.camera.placement.pos};
  float &zoom {meow::app.camera.zoom};
  pos = (pos / zoom) + cam;

  for (bicudo::physics::placement *&p_placements : meow::app.bicudo.placement_list) {
    if (bicudo::aabb_collide_with_vec2(p_placements->min, p_placements->max, pos)) {
      p_placement = p_placements;
      return true;
    }
  }

  return false;
}

void meow::tools_pick_camera(
  meow::tools::pickup_info_t *p_pickup_info
) {
  meow::camera &camera {meow::app.camera};
  ekg::vec4 &interact {ekg::input::interact()};

  if (ekg::hovered.id == 0 && ekg::input::action("zoom-camera")) {
    meow::app.camera.interpolated_zoom = bicudo_clamp_min(
      meow::app.camera.interpolated_zoom + interact.w * 0.09f,
      0.000001f
    );
  }

  if (!p_pickup_info->p_placement && ekg::hovered.id == 0 && ekg::input::action("click-on-camera")) {
    p_pickup_info->p_placement = &camera.placement;

    p_pickup_info->delta.x = interact.x - p_pickup_info->p_placement->min.x;
    p_pickup_info->delta.y = interact.y - p_pickup_info->p_placement->min.y;

    p_pickup_info->pick_pos.x = p_pickup_info->p_placement->pos.x;
    p_pickup_info->pick_pos.y = p_pickup_info->p_placement->pos.y;

    p_pickup_info->prev_pos.x = interact.x;
    p_pickup_info->prev_pos.y = interact.y;

    meow::app.immediate.latest_pos_clicked = {interact.x, interact.y};
  } else if (ekg::input::action("drop-camera")) {
    p_pickup_info->p_placement = nullptr;
  }
}

void meow::tools_update_picked_camera(
  meow::tools::pickup_info_t *p_pickup_info
) {
  if (meow::app.camera.interpolated_zoom != meow::app.camera.zoom) {
    meow::app.camera.zoom = bicudo::lerp<float>(
      meow::app.camera.zoom,
      meow::app.camera.interpolated_zoom,
      bicudo::dt
    );

    meow::app.immediate.set_viewport(
      meow::app.immediate.viewport.z,
      meow::app.immediate.viewport.w
    );
  }

  meow::app.camera.on_update();

  if (!p_pickup_info->p_placement) {
    return;
  }

  ekg::vec4 &interact {ekg::input::interact()};

  p_pickup_info->p_placement->velocity = {
    -((interact.x - p_pickup_info->delta.x) - (p_pickup_info->prev_pos.x - p_pickup_info->delta.x)),
    -((interact.y - p_pickup_info->delta.y) - (p_pickup_info->prev_pos.y - p_pickup_info->delta.y))
  };

  p_pickup_info->prev_pos.x = interact.x;
  p_pickup_info->prev_pos.y = interact.y;
}

bicudo::collided meow::tools_pick_object_from_world(
  tools::pickup_info_t *p_pickup_info
) {
  ekg::vec4 &interact {ekg::input::interact()};

  if (
      !p_pickup_info->p_placement &&
      ekg::hovered.id == 0 &&
      ekg::input::action("click-on-object") &&
      meow::tools_pick_physics_placement(p_pickup_info->p_placement, bicudo::vec2(interact.x, interact.y))
    ) {

    p_pickup_info->delta.x = interact.x - p_pickup_info->p_placement->min.x;
    p_pickup_info->delta.y = interact.y - p_pickup_info->p_placement->min.y;

    p_pickup_info->pick_pos.x = p_pickup_info->p_placement->pos.x;
    p_pickup_info->pick_pos.y = p_pickup_info->p_placement->pos.y;

    p_pickup_info->prev_pos.x = interact.x;
    p_pickup_info->prev_pos.y = interact.y;

    meow::tools_to_local_camera(&p_pickup_info->pick_pos);
    meow::tools_to_local_camera(&p_pickup_info->prev_pos);
    meow::tools_to_local_camera(&p_pickup_info->delta);

    return true;
  } else if (ekg::input::action("drop-object")) {
    p_pickup_info->p_placement = nullptr;
    return false;
  }

  return false;
}

void meow::tools_update_picked_object(
  meow::tools::pickup_info_t *p_pickup_info
) {
  if (!p_pickup_info->p_placement) {
    return;
  }

  ekg::vec4 interact {ekg::input::interact()};

  interact.x /= meow::app.camera.zoom;
  interact.y /= meow::app.camera.zoom;

  p_pickup_info->p_placement->velocity = {
    ((interact.x - p_pickup_info->delta.x) - (p_pickup_info->prev_pos.x - p_pickup_info->delta.x)),
    ((interact.y - p_pickup_info->delta.y) - (p_pickup_info->prev_pos.y - p_pickup_info->delta.y))
  };

  if (bicudo::assert_float(p_pickup_info->p_placement->mass, 0.0f)) {
    bicudo::physics_placement_move(
      p_pickup_info->p_placement,
      p_pickup_info->p_placement->velocity
    );

    p_pickup_info->p_placement->velocity = {};
  }

  p_pickup_info->prev_pos.x = interact.x;
  p_pickup_info->prev_pos.y = interact.y;
}