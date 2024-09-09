#ifndef BICUDO_WORLD_PHYSICS_SIMULATOR_HPP
#define BICUDO_WORLD_PHYSICS_SIMULATOR_HPP

#include "bicudo/util/math.hpp"
#include <vector>

namespace bicudo::world::physics {
  struct simulator {
  public:
    std::vector<bicudo::placement*> placement_list {};
  };
}

namespace bicudo {
  void world_physics_update_simulator(bicudo::world::physics::simulator *p_simulator);
}

#endif
