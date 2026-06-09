#ifndef BICUDO_CORE_HPP
#define BICUDO_CORE_HPP

#include <bicudo/pipeline/base.hpp>

namespace bicudo {
  struct init_core_t {
  public:
    bicudo::pipeline::base *p_base;
  };

  struct core_t {
  public:
    bicudo::pipeline::base *p_base;
  };

  extern bicudo::core_t *p_core;
}

#endif
