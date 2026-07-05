#ifndef BICUDO_GPU_HPP
#define BICUDO_GPU_HPP

#include <cstdint>
#include <bicudo/io/atomic.hpp>
#include <vector>

namespace bicudo {
  enum hypergroup_state {
    IDLE, // idle intentional for development
    UNHANDLED,
    MUST_REFRESH_HOST,
    MUST_DISPATCH,
    MUST_WAIT,
    MUST_ASYNC,
    MUST_REFILL
  };

  using hypergroup_state_t = hypergroup_state;

  struct gpu_sacred_atomic_memory_t {
  public:
    void *p_host {};
    void *p_device {};
    std::size_t bytes {};
    bicudo::sacred_atomic_state_t state = bicudo::sacred_atomic_state::HOST_FREE;
  };

  struct gpu_sacred_arg_t {
  public:
    std::string tag {};
    void *p_device {};
    std::size_t bytes {};
  };

  using gpu_sacred_arguments_t = std::vector<bicudo::gpu_sacred_arg_t>;
  using gpu_sacred_param_pointers_t = std::vector<void*>;

  struct gpu_sacred_dispatch_params_t {
  public:
    bicudo::gpu_sacred_param_pointers_t *p_pointers {};
    std::size_t length {};
  };
}

#endif
