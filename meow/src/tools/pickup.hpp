#ifndef MEOW_TOOLS_PICKUP_HPP
#define MEOW_TOOLS_PICKUP_HPP

#include "bicudo/world/object.hpp"

namespace meow::tools {
  struct pickup_info_t {
  public:
    bicudo::vec2 delta {};
    bicudo::vec2 pick_pos {};
    bicudo::vec2 prev_pos {};
    bicudo::placement *p_placement {};
    bicudo::object *p_obj {};
  };
}

namespace meow {
  void tools_pick_camera(
    meow::tools::pickup_info_t *p_pickup_info
  );

  void tools_update_picked_camera(
    meow::tools::pickup_info_t *p_pickup_info
  );

  bicudo::collided tools_pick_object_from_world(
    meow::tools::pickup_info_t *p_pickup_info
  );

  void tools_update_picked_object(
    meow::tools::pickup_info_t *p_pickup_info
  );
}

#endif