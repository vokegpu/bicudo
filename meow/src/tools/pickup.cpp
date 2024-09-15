#include "pickup.hpp"

#include "bicudo/bicudo.hpp"
#include <ekg/ekg.hpp>

bicudo::collided meow::tools_pick_object_from_world(
  tools::pickup_info_t *p_pickup_info
) {
  ekg::vec4 &interact {ekg::input::interact()};

  if (!p_pickup_info->p_obj && ekg::hovered.id == 0 && ekg::input::action("click-on-object") && bicudo::world::pick(p_pickup_info->p_obj, {interact.x, interact.y})) {
    p_pickup_info->delta.x = interact.x - p_pickup_info->p_obj->placement.min.x;
    p_pickup_info->delta.y = interact.y - p_pickup_info->p_obj->placement.min.y;

    p_pickup_info->pick_pos.x = p_pickup_info->p_obj->placement.pos.x;
    p_pickup_info->pick_pos.y = p_pickup_info->p_obj->placement.pos.y;

    p_pickup_info->prev_pos.x = interact.x;
    p_pickup_info->prev_pos.y = interact.y;
    
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

  ekg::vec4 &interact {ekg::input::interact()};

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