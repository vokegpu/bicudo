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
    object(bicudo::placement);

    virtual void on_low_latency_update() {};
    virtual void on_update();
  };
}

#endif