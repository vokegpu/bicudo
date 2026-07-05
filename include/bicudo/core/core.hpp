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

namespace bicudo {
  template<typename t>
  t &as_gpu() {
    return *dynamic_cast<t*>(bicudo::p_core->p_base);
  }

  std::string device();
}

#endif
