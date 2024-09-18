#include "pickup.hpp"

#include "bicudo/bicudo.hpp"
#include <ekg/ekg.hpp>
#include "meow.hpp"

void meow::tools_pick_camera(
  meow::tools::pickup_info_t *p_pickup_info
) {
  bicudo::camera &camera {bicudo::world::camera()};
  ekg::vec4 &interact {ekg::input::interact()};

  if (ekg::hovered.id == 0 && ekg::input::action("zoom-camera")) {
    bicudo::app.world_manager.camera.interpolated_zoom = bicudo_clamp_min(
      bicudo::app.world_manager.camera.interpolated_zoom + interact.w * 0.09f,
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
  if (bicudo::app.world_manager.camera.interpolated_zoom != bicudo::app.world_manager.camera.zoom) {
    bicudo::app.world_manager.camera.zoom = bicudo::lerp<float>(
      bicudo::app.world_manager.camera.zoom,
      bicudo::app.world_manager.camera.interpolated_zoom,
      bicudo::dt
    );

    meow::app.immediate.set_viewport(
      meow::app.immediate.viewport.z,
      meow::app.immediate.viewport.w
    );
  }

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
      !p_pickup_info->p_obj &&
      ekg::hovered.id == 0 &&
      ekg::input::action("click-on-object") &&
      bicudo::world::pick(p_pickup_info->p_obj, bicudo::vec2(interact.x, interact.y))
    ) {

    p_pickup_info->delta.x = interact.x - p_pickup_info->p_obj->placement.min.x;
    p_pickup_info->delta.y = interact.y - p_pickup_info->p_obj->placement.min.y;

    p_pickup_info->pick_pos.x = p_pickup_info->p_obj->placement.pos.x;
    p_pickup_info->pick_pos.y = p_pickup_info->p_obj->placement.pos.y;

    p_pickup_info->prev_pos.x = interact.x;
    p_pickup_info->prev_pos.y = interact.y;

    bicudo::to_local_camera(&p_pickup_info->pick_pos);
    bicudo::to_local_camera(&p_pickup_info->prev_pos);
    bicudo::to_local_camera(&p_pickup_info->delta);

    return true;
  } else if (ekg::input::action("drop-object")) {
    p_pickup_info->p_obj = nullptr;
    return false;
  }

  return false;
}

void meow::tools_update_picked_object(
  meow::tools::pickup_info_t *p_pickup_info
) {
  if (!p_pickup_info->p_obj) {
    return;
  }

  ekg::vec4 interact {ekg::input::interact()};

  interact.x /= bicudo::app.world_manager.camera.zoom;
  interact.y /= bicudo::app.world_manager.camera.zoom;

  p_pickup_info->p_obj->placement.velocity = {
    ((interact.x - p_pickup_info->delta.x) - (p_pickup_info->prev_pos.x - p_pickup_info->delta.x)),
    ((interact.y - p_pickup_info->delta.y) - (p_pickup_info->prev_pos.y - p_pickup_info->delta.y))
  };

  if (bicudo::assert_float(p_pickup_info->p_obj->placement.mass, 0.0f)) {
    bicudo::move(
      &p_pickup_info->p_obj->placement,
      p_pickup_info->p_obj->placement.velocity
    );

    p_pickup_info->p_obj->placement.velocity = {};
  }

  p_pickup_info->prev_pos.x = interact.x;
  p_pickup_info->prev_pos.y = interact.y;
}