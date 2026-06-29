#ifndef BICUDO_GPU_HPP
#define BICUDO_GPU_HPP

#include <bicudo/core/core.hpp>

namespace bicudo {
  template<typename t>
  t &as_gpu() {
    return *dynamic_cast<t*>(bicudo::p_core->p_base);
  }
}

#endif
