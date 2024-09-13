#ifndef CLIENT_TOOLS_PICKUP_HPP
#define CLIENT_TOOLS_PICKUP_HPP

#include "bicudo/world/object.hpp"

namespace client::tools {
  struct pickup_info_t {
  public:
    bicudo::vec2 delta {};
    bicudo::vec2 pick_pos {};
    bicudo::vec2 prev_pos {};
    bicudo::object *p_obj {};
  };
}

namespace client {
  bicudo::collided tools_pick_object_from_world(
    client::tools::pickup_info_t *p_pickup_info
  );

  void tools_update_picked_object(
    client::tools::pickup_info_t *p_pickup_info
  );
}

#endif