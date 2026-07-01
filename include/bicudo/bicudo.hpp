#ifndef BICUDO_HPP
#define BICUDO_HPP

#include <bicudo/core/core.hpp>
#include <bicudo/physics/physics.hpp>
#include <bicudo/log/log.hpp>

namespace bicudo {
  bicudo::result_t init(
    bicudo::init_core_t &init_core,
    bicudo::core_t &core
  );
}

#endif
