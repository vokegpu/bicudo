#ifndef BICUDO_GPU_ROCM_SACRED_HPP
#define BICUDO_GPU_ROCM_SACRED_HPP

#include <cstdint>
#include <vector>

#include <bicudo/gpu/rocm/divine.hpp>

namespace bicudo {
  struct gpu_rm_sacred_atomic_memory_t {
  public:
    void *p_host {};
    void *p_device {};
    void *p_out {};
    std::size_t bytes {};
  public:
    gpu_rm_sacred_atomic_memory_t(std::size_t bytes) {
      this->bytes = bytes;
    }

    gpu_rm_sacred_atomic_memory_t() = default;
  };

  bicudo::result_t gpu_allocate_sacred_atomic(
    bicudo::gpu_rm_sacred_atomic_memory_t &atomic,
    std::size_t hip_host_malloc_flags,
    std::size_t hip_host_get_device_pointer_flags
  );

  bicudo::result_t gpu_sacred_async_fetch(
    bicudo::gpu_rm_divine_fun_t &fun,
    void *p_host,
    void *p_device,
    std::size_t bytes
  );

  bicudo::result_t gpu_sacred_call(
    bicudo::gpu_rm_divine_module_t &kmodule,
    bicudo::gpu_rm_divine_fun_t &fun
  );
}

#endif
