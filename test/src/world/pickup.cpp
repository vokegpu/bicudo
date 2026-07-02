#include "meow.hpp"
#include "pickup.hpp"
#include <bicudo/bicudo.hpp>
#include <ekg/ekg.hpp>

void meow::tools_to_local_camera(bicudo::vec2_t<float> &vec) {
  vec.x /= meow::app.camera.zoom;
  vec.y /= meow::app.camera.zoom;
}

bool meow::tools_pick_physics_body(
  bicudo::hypergroup_t &hypergroup,
  bicudo::body_t **p_body,
  bicudo::vec2_t<float> pos
) {
  bicudo::vec2_t<float> &cam {meow::app.camera.rect.pos};
  float &zoom {meow::app.camera.zoom};
  pos = (pos / zoom) + cam;

  meow::app.gui.has_some_body_click = false;
  for (bicudo::body_t *p_b : hypergroup.bodies) {
    if (bicudo::aabb_collide_with_vec2(p_b->min, p_b->max, pos)) {
      *p_body = p_b;
      meow::app.gui.has_some_body_click = true;
      return true;
    }
  }

  return false;
}

void meow::tools_pick_camera(
  meow::pickup_info_t &pickup_info
) {
  meow::camera &camera {meow::app.camera};
  ekg::vec4_t<float> &interact {ekg::input().interact};

  if (ekg::gui.ui.hovered_type == ekg::type::unknown && ekg::fired("zoom-camera")) {
    meow::app.camera.set_zoom(
      bicudo_clamp_min(
        meow::app.camera.zoom + interact.w * 0.09f,
        0.000001f
      )
    );
  }

  if (
    !pickup_info.p_body
    &&
    ekg::gui.ui.hovered_type == ekg::type::unknown
    &&
    ekg::fired("click-on-camera")
    &&
    !meow::app.gui.has_some_body_click
  ) {
    pickup_info.p_body = &camera.rect;    

    pickup_info.delta.x = interact.x - pickup_info.p_body->min.x;
    pickup_info.delta.y = interact.y - pickup_info.p_body->min.y;

    pickup_info.pick_pos.x = pickup_info.p_body->pos.x;
    pickup_info.pick_pos.y = pickup_info.p_body->pos.y;

    pickup_info.prev_pos.x = interact.x;
    pickup_info.prev_pos.y = interact.y;

    meow::app.immediate.latest_pos_clicked = {interact.x, interact.y};
  } else if (ekg::fired("drop-camera")) {
    pickup_info.p_body = nullptr;
  }

  if (!meow::app.gui.has_some_body_click && ekg::fired("world-popup")) {
    ekg::show(meow::app.gui.in_world_popup, interact);
  }
}

void meow::tools_update_picked_camera(
  meow::pickup_info_t &pickup_info
) {
  meow::app.camera.on_update();

  if (meow::app.camera.is_while_zoom) {
    meow::app.immediate.set_viewport(
      meow::app.immediate.viewport.z,
      meow::app.immediate.viewport.w
    );
  }

  if (!pickup_info.p_body) {
    return;
  }

  ekg::vec4_t<float> &interact {ekg::input().interact};

  pickup_info.p_body->velocity = {
    -((interact.x - pickup_info.delta.x) - (pickup_info.prev_pos.x - pickup_info.delta.x)),
    -((interact.y - pickup_info.delta.y) - (pickup_info.prev_pos.y - pickup_info.delta.y))
  };

  pickup_info.prev_pos.x = interact.x;
  pickup_info.prev_pos.y = interact.y;
}

bool meow::tools_pick_object_from_world(
  bicudo::hypergroup_t &hypergroup,
  meow::pickup_info_t &pickup_info
) {
  ekg::vec4_t<float> &interact {ekg::input().interact};

  if (ekg::gui.ui.hovered_type != ekg::type::unknown) {
    return false;
  }

  bool should_move_object {
    ekg::fired("move-object")
  };

  bool should_options_object {
    ekg::fired("options-object")
  };

  bool find {
    (
      (!pickup_info.p_body || pickup_info.action == meow::action::PICKUP_OPTIONS_OBJ)
      &&
      should_move_object
    )
    ||
    (
      (!pickup_info.p_body || pickup_info.action == meow::action::PICKUP_OPTIONS_OBJ)
      &&
      should_options_object
    )
  };

  if (
    find
    &&
    !meow::tools_pick_physics_body(
      hypergroup,
      &pickup_info.p_body,
      bicudo::vec2_t<float>(interact.x, interact.y)
    )
  ) {
    return false;
  }

  if (
      find
      &&
      should_move_object
    ) {
    
    pickup_info.p_body->no_gravity = true;
    pickup_info.delta.x = interact.x - pickup_info.p_body->min.x;
    pickup_info.delta.y = interact.y - pickup_info.p_body->min.y;

    pickup_info.pick_pos.x = pickup_info.p_body->pos.x;
    pickup_info.pick_pos.y = pickup_info.p_body->pos.y;

    pickup_info.prev_pos.x = interact.x;
    pickup_info.prev_pos.y = interact.y;

    meow::tools_to_local_camera(pickup_info.pick_pos);
    meow::tools_to_local_camera(pickup_info.prev_pos);
    meow::tools_to_local_camera(pickup_info.delta);

    pickup_info.action = meow::action::PICKUP_MOVE_OBJ;
    return true;
  } else if (pickup_info.p_body && ekg::fired("drop-object")) {
    pickup_info.p_body->no_gravity = false;
    pickup_info.p_body = nullptr;
    pickup_info.action = meow::action::NONE;
    return false;
  } else if (
    find
    &&
    should_options_object
  ) {
    pickup_info.action = meow::action::PICKUP_OPTIONS_OBJ;
  }

  return false;
}

void meow::tools_update_picked_object(
  bicudo::hypergroup_t &hypergroup,
  meow::pickup_info_t &pickup_info
) {
  if (!pickup_info.p_body || pickup_info.action != meow::action::PICKUP_MOVE_OBJ) {
    return;
  }

  ekg::vec4_t<float> interact {ekg::input().interact};

  interact.x /= meow::app.camera.zoom;
  interact.y /= meow::app.camera.zoom;

  pickup_info.p_body->velocity = {
    ((interact.x - pickup_info.delta.x) - (pickup_info.prev_pos.x - pickup_info.delta.x)),
    ((interact.y - pickup_info.delta.y) - (pickup_info.prev_pos.y - pickup_info.delta.y))
  };

  if (bicudo::assert_float(pickup_info.p_body->mass, 0.0f)) {
    bicudo::update(
      &hypergroup,
      pickup_info.p_body
    );

    pickup_info.p_body->velocity = {};
  }

  pickup_info.prev_pos.x = interact.x;
  pickup_info.prev_pos.y = interact.y;
}
