#ifndef BICUDO_GPU_ROCM_SACRED_HPP
#define BICUDO_GPU_ROCM_SACRED_HPP

#include <cstdint>
#include <vector>

#include <bicudo/gpu/gpu.hpp>
#include <bicudo/gpu/rocm/divine.hpp>

namespace bicudo {
  bicudo::result_t gpu_sacred_allocate_atomic(
    bicudo::gpu_sacred_atomic_memory_t &atomic,
    std::size_t hip_host_malloc_flags,
    std::size_t hip_host_get_device_pointer_flags
  );

  bicudo::result_t gpu_sacred_reallocate_atomic(
    bicudo::gpu_sacred_atomic_memory_t &atomic,
    std::size_t new_bytes,
    std::size_t hip_host_malloc_flags,
    std::size_t hip_host_get_device_pointer_flags
  );

  bicudo::result_t gpu_sacred_async_fetch(
    bicudo::gpu_rm_divine_memory_t &memory,
    void *p_host,
    void *p_device,
    std::size_t bytes
  );

  bicudo::result_t gpu_free_sacred_atomic(
    bicudo::gpu_sacred_atomic_memory_t &atomic
  );

  bicudo::result_t gpu_sacred_call(
    bicudo::gpu_rm_divine_module_t &kmodule,
    bicudo::gpu_rm_divine_fun_t &fun,
    bicudo::gpu_rm_divine_dispatch_properties_t &properties
  );

  void gpu_sacred_fetch_dispatch_params(
    bicudo::gpu_rm_divine_dispatch_properties_t &properties,
    bicudo::gpu_sacred_dispatch_params_t &params,
    bicudo::gpu_sacred_arguments_t &arguments,
    bicudo::gpu_sacred_param_pointers_t &pointers
  );
}

#endif
