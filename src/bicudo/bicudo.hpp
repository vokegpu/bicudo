#ifndef BICUDO_HPP
#define BICUDO_HPP

#include "bicudo/util/log.hpp"
#include "bicudo/world/management/world_manager.hpp"
#include <cstdint>

namespace bicudo {
  extern struct application {
  public:
    bicudo::world_manager world_manager {};
  } app;

  void count(uint32_t *p_number);
}

#endif