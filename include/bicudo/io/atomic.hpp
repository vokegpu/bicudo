#ifndef BICUDO_IO_ATOMIC_HPP
#define BICUDO_IO_ATOMIC_HPP

#include <cstdint>
#include <bicudo/math/geometry.hpp>

namespace bicudo {
  enum class sacred_atomic_state {
    HOST_FREE,
    HOST_ALLOCATED
  };

  using sacred_atomic_state_t = bicudo::sacred_atomic_state;
}

namespace bicudo {
  template<typename t>
  void pass_memory(
    void *p_src,
    std::size_t &i,
    t value
  ) {
    t *p_casted_src = static_cast<t*>(p_src);
    p_casted_src[i++] = value;
  }

  template<typename t>
  void pass_memory(
    void *p_src,
    std::size_t &i,
    bicudo::vec2_t<t> value
  ) {
    bicudo::pass_memory<t>(p_src, i, value.x);
    bicudo::pass_memory<t>(p_src, i, value.y);
  }
}

#endif
