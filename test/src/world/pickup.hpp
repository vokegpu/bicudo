#ifndef MEOW_TOOLS_PICKUP_HPP
#define MEOW_TOOLS_PICKUP_HPP

#include <bicudo/physics/physics.hpp>

namespace meow {
  enum action {
    NONE,
    PICKUP_MOVE_OBJ,
    PICKUP_OPTIONS_OBJ
  };

  struct pickup_info_t {
  public:
    bicudo::vec2_t<float> delta {};
    bicudo::vec2_t<float> pick_pos {};
    bicudo::vec2_t<float> prev_pos {};
    bicudo::body_t *p_body {};
    meow::action action {};
  };
}

namespace meow {
  void tools_to_local_camera(
    bicudo::vec2_t<float> &vec
  );

  bool tools_pick_physics_body(
    bicudo::hypergroup_t &hypergroup,
    bicudo::body_t **p_body,
    bicudo::vec2_t<float> pos
  );

  void tools_pick_camera(
    meow::pickup_info_t &pickup_info
  );

  void tools_update_picked_camera(
    meow::pickup_info_t &pickup_info
  );

  bool tools_pick_object_from_world(
    bicudo::hypergroup_t &hypergroup,
    meow::pickup_info_t &pickup_info
  );

  void tools_update_picked_object(
    bicudo::hypergroup_t &hypergroup,
    meow::pickup_info_t &pickup_info
  );
}

#endif
