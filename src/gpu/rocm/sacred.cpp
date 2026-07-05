#include <bicudo/gpu/rocm/sacred.hpp>
#include <bicudo/log/log.hpp>

bicudo::result_t bicudo::gpu_free_sacred_atomic(
  bicudo::gpu_sacred_atomic_memory_t &atomic
) {
  hipError_t e {};
  bicudo_assert(
    (
      e = hipHostFree(
        atomic.p_host
      )
    ),
    hipSuccess,
    bicudo::loge("Failed to free sacred atomic memory.")
  );

  atomic.state = bicudo::sacred_atomic_state_t::HOST_FREE;
  bicudo::log("Sacred atomic memory back to the Pleroma.");
  return e == hipSuccess ? bicudo::result::SUCCESS : bicudo::result::FAILED_TO_FREE_ATOMIC_MEMORY;
}

bicudo::result_t bicudo::gpu_sacred_async_fetch(
  bicudo::gpu_rm_divine_memory_t &memory,
  void *p_host,
  void *p_device,
  std::size_t bytes
) {
  hipError_t e {};
  bicudo_assert(
    (
      e = hipMemcpyDtoHAsync(
        p_host,
        p_device,
        bytes,
        memory.hip_stream
      )
    ),
    hipSuccess,
    bicudo::loge("Failed to async fetch sacred atomic memory.")
  );

  return e == hipSuccess ? bicudo::result::SUCCESS : bicudo::result::FAILED_TO_ASYNC_FETCH_ATOMIC_MEMORY;
}

bicudo::result_t bicudo::gpu_sacred_allocate_atomic(
  bicudo::gpu_sacred_atomic_memory_t &atomic,
  std::size_t hip_host_malloc_flags,
  std::size_t hip_host_get_device_pointer_flags
) {
  bicudo_assert(
    hipHostMalloc(
      &atomic.p_host,
      atomic.bytes,
      hip_host_malloc_flags
    ),
    hipSuccess,
    bicudo::loge("Failed to allocate sagred atomic bytes ", atomic.bytes)
  );

  bicudo_assert(
    hipHostGetDevicePointer(
      &atomic.p_device,
      atomic.p_host,
      hip_host_get_device_pointer_flags
    ),
    hipSuccess,
    bicudo::loge("Failed to allocate sagred atomic bytes ", atomic.bytes)
  );

  atomic.state = bicudo::sacred_atomic_state_t::HOST_ALLOCATED;
  return bicudo::result::SUCCESS;
}

bicudo::result_t bicudo::gpu_sacred_reallocate_atomic(
  bicudo::gpu_sacred_atomic_memory_t &atomic,
  std::size_t new_bytes,
  std::size_t hip_host_malloc_flags,
  std::size_t hip_host_get_device_pointer_flags
) {
  switch (atomic.state) {
  case bicudo::sacred_atomic_state::HOST_FREE:
    atomic.bytes = new_bytes;
    return bicudo::gpu_sacred_allocate_atomic(
      atomic,
      hip_host_malloc_flags,
      hip_host_get_device_pointer_flags
    );
  case bicudo::sacred_atomic_state::HOST_ALLOCATED:
    bicudo::gpu_sacred_atomic_memory_t new_atomic {
      .bytes = new_bytes
    };

    bicudo::result_t result = bicudo::gpu_sacred_allocate_atomic(
      new_atomic,
      hip_host_malloc_flags,
      hip_host_get_device_pointer_flags
    );

    if (result == bicudo::FAILED_TO_ALLOCATE_HOST_MEMORY) {
      return result;
    }

    std::size_t diff {
      new_bytes < atomic.bytes
        ? new_bytes : atomic.bytes
    };

    hipError_t err = hipMemcpy(
      new_atomic.p_device,
      atomic.p_host,
      diff,
      hipMemcpyHostToHost
    );

    if (err != hipSuccess) {
      bicudo::loge(
        "Could not memory copy from old host (", atomic.bytes, ") to new host (", new_bytes, ")"
      );
      return bicudo::result::FAILED_TO_SYNC_FETCH_ATOMIC_MEMORY;
    }

    err = hipHostFree(atomic.p_host);
    if (err != hipSuccess) {
      bicudo::loge("Could not free atomic memory.");
      return bicudo::result::FAILED_TO_FREE_ATOMIC_MEMORY;
    }

    atomic.p_host = new_atomic.p_host;
    atomic.p_device = new_atomic.p_device;
    atomic.bytes = new_atomic.bytes;
    atomic.state = bicudo::sacred_atomic_state::HOST_FREE;

    return bicudo::result::SUCCESS;
  }

  return result::OK;
}

bicudo::result_t bicudo::gpu_sacred_call(
  bicudo::gpu_rm_divine_module_t &kmodule,
  bicudo::gpu_rm_divine_fun_t &fun,
  bicudo::gpu_rm_divine_dispatch_properties_t &properties
) {
  void *pv_configs[] {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, properties.memory.params.p_pointers->data(),
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &properties.memory.params.length,
    HIP_LAUNCH_PARAM_END
  };

  hipError_t result {};
  bicudo_assert(
    (
      result = (
        hipModuleLaunchKernel(
          fun.entry_point.h_fun,
          properties.dimension.grid.x,
          properties.dimension.grid.y,
          properties.dimension.grid.z,
          properties.dimension.block.x,
          properties.dimension.block.y,
          properties.dimension.block.z,
          properties.memory.shared_mem_bytes,
          properties.memory.hip_stream,
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

void bicudo::gpu_sacred_fetch_dispatch_params(
  bicudo::gpu_rm_divine_dispatch_properties_t &properties,
  bicudo::gpu_sacred_dispatch_params_t &params,
  bicudo::gpu_sacred_arguments_t &arguments,
  bicudo::gpu_sacred_param_pointers_t &pointers
) {
  params.p_pointers = &pointers;
  pointers.clear();

  for (bicudo::gpu_sacred_arg_t &arg : arguments) {
    pointers.push_back(arg.p_device);
    params.length += arg.bytes;
  }

  properties.memory.params = params;
}
