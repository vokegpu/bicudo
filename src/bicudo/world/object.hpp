#ifndef BICUDO_WORLD_OBJECT_HPP
#define BICUDO_WORLD_OBJECT_HPP

#include "types.hpp"
#include "bicudo/util/math.hpp"

namespace bicudo {
  class object {
  public:
    bicudo::placement placement {};
    bicudo::id id {};
  public:
    virtual void low_latency_update() {};
    // virtual void update() {};
  };
}

#endif