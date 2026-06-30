#include <bicudo/gpu/rocm/sacred.hpp>
#include <bicudo/log/log.hpp>

bicudo::result_t bicudo::gpu_free_sacred_atomic(
  bicudo::gpu_rm_sacred_atomic_memory_t &atomic
) {
  hipError_t e {};
  bicudo_hip_assert(
    (
      e = hipHostFree(
        atomic.p_host
      )
    ),
    hipSuccess,
    bicudo::loge("Failed to free sacred atomic memory.")
  );

  return e == hipSuccess ? bicudo::result::SUCCESS : bicudo::result::FAILED_TO_FREE_ATOMIC_MEMORY;
}

bicudo::result_t bicudo::gpu_sacred_async_fetch(
  bicudo::gpu_rm_divine_fun_t &fun,
  void *p_host,
  void *p_device,
  std::size_t bytes
) {
  hipError_t e {};
  bicudo_hip_assert(
    (
      e = hipMemcpyDtoHAsync(
        p_host,
        p_device,
        bytes,
        fun.memory.h_stream
      )
    ),
    hipSuccess,
    bicudo::loge(bicudo::logtf(fun), "Failed to async fetch sacred atomic memory.")
  );

  return e == hipSuccess ? bicudo::result::SUCCESS : bicudo::result::FAILED_TO_ASYNC_FETCH_ATOMIC_MEMORY;
}

bicudo::result_t bicudo::gpu_allocate_sacred_atomic(
  bicudo::gpu_rm_sacred_atomic_memory_t &atomic,
  std::size_t hip_host_malloc_flags,
  std::size_t hip_host_get_device_pointer_flags
) {
  bicudo_hip_assert(
    hipHostMalloc(
      &atomic.p_host,
      atomic.bytes,
      hip_host_malloc_flags
    ),
    hipSuccess,
    bicudo::loge("Failed to allocate sagred atomic bytes ", atomic.bytes)
  );

  bicudo_hip_assert(
    hipHostGetDevicePointer(
      &atomic.p_device,
      atomic.p_host,
      hip_host_get_device_pointer_flags
    ),
    hipSuccess,
    bicudo::loge("Failed to allocate sagred atomic bytes ", atomic.bytes)
  );

  return bicudo::result::OK;
}

bicudo::result_t bicudo::gpu_sacred_call(
  bicudo::gpu_rm_divine_module_t &kmodule,
  bicudo::gpu_rm_divine_fun_t &fun
) {
  void *pv_configs[] {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, fun.memory.sacred_pointers.data(),
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &fun.memory.sacred_pointers_mem_bytes_length,
    HIP_LAUNCH_PARAM_END
  };

  hipError_t result {};
  bicudo_hip_assert(
    (
      result = (
        hipModuleLaunchKernel(
          fun.entry_point.h_fun,
          fun.dimension.grid.x,
          fun.dimension.grid.y,
          fun.dimension.grid.z,
          fun.dimension.block.x,
          fun.dimension.block.y,
          fun.dimension.block.z,
          fun.memory.shared_mem_bytes,
          fun.memory.h_stream,
          nullptr,
          pv_configs
        )
      )
    ),
    hipSuccess,
    bicudo::loge("Module '", kmodule.tag, "' has failed to call entry-point '", fun.entry_point.name, "'.")
  );

  return result == hipSuccess ? bicudo::result::OK : bicudo::result::FAILED_TO_CALL_FUNCTION;
}
