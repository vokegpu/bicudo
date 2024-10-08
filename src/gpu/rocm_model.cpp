#include "bicudo/gpu/rocm_model.hpp"
#include "bicudo/util/log.hpp"

bicudo::result bicudo::gpu_rocm_dispatch(
  bicudo::gpu::rocm_pipeline *p_pipeline,
  uint64_t module_index,
  uint64_t kernel_index
) {
  bicudo::gpu::rocm_kernel &kernel {p_pipeline->kernel_list.at(module_index)};
  bicudo::gpu::rocm_function &function {kernel.function_list.at(kernel_index)};

  void *p_configs[] {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, function.argument_list.data(),
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &function.mem_size,
    HIP_LAUNCH_PARAM_END
  };

  hipError_t err {
    hipModuleLaunchKernel(
      function.entry_point,
      function.grid.x,
      function.grid.y,
      function.grid.z,
      function.block.x,
      function.block.y,
      function.block.z,
      function.shared_mem_bytes,
      function.stream,
      nullptr,
      p_configs
    )
  };

  if (err != hipSuccess) {
    bicudo::log()
      << "[GPU] failed to dispatch computer kernel '"
      << kernel.p_tag
      << "':\n" 
      << hipGetErrorName(err);
    return bicudo::types::FAILED;
  }

  return bicudo::types::SUCCESS;
}

bicudo::result bicudo::gpu_rocm_memory_fetch(
  bicudo::gpu::rocm_pipeline *p_pipeline,
  uint64_t module_index,
  uint64_t kernel_index,
  uint64_t param_index,
  bicudo::types op_type
) {
  bicudo::gpu::rocm_kernel &kernel {p_pipeline->kernel_list.at(module_index)};
  bicudo::gpu::rocm_function &function {kernel.function_list.at(kernel_index)};
  bicudo::gpu::buffer_t &buffer {function.buffer_list.at(param_index)};

  void *p_dst {};
  void *p_src {};
  hipMemcpyKind mem_copy_kind {};

  switch (op_type) {
  case bicudo::types::WRITEBACK:
    p_dst = buffer.p_host;
    p_src = buffer.p_device;
    mem_copy_kind = hipMemcpyDeviceToHost;
    break;
  case bicudo::types::WRITESTORE:
    p_dst = buffer.p_device;
    p_src = buffer.p_host;
    mem_copy_kind = hipMemcpyHostToDevice;
    break;
  }

  hip_validate(
    hipMemcpy(
      p_dst,
      p_src,
      buffer.size,
      mem_copy_kind
    ),
    "failed to copy memory"
  );

  return bicudo::types::SUCCESS;
}

bicudo::result bicudo::gpu_rocm_create_pipeline(
  bicudo::gpu::rocm_pipeline *p_pipeline,
  bicudo::gpu::rocm_pipeline_create_info *p_pipeline_create_info
) {
  p_pipeline->p_tag = p_pipeline_create_info->p_tag;
  p_pipeline->kernel_list = p_pipeline_create_info->kernel_list;

  bicudo::result result {bicudo::types::SUCCESS};
  std::vector<char> binary_list {};

  for (bicudo::gpu::rocm_kernel &kernel : p_pipeline->kernel_list) {
    hiprtc_validate(
      hiprtcCreateProgram(
        &kernel.program,
        kernel.p_src,
        kernel.p_tag,
        0,
        nullptr,
        nullptr
      ),
      "failed to create program"
    );

    hiprtc_validate(
      hiprtcCompileProgram(
        kernel.program,
        0,
        nullptr
      ),
      "failed to compile program"
    );

    size_t log_size {};
    hiprtc_validate(
      hiprtcGetProgramLogSize(
        kernel.program,
        &log_size
      ),
      "failed to get program log size"
    );

    if (log_size) {
      std::string log {}; log.resize(log_size);
      hiprtc_validate(
        hiprtcGetProgramLog(
          kernel.program,
          log.data()
        ),
        "failed to get program log"
      );

      bicudo::log() << "failed to create program, more info:\n" << log;
      result = bicudo::types::FAILED;
      continue;
    }

    size_t kernel_binary_size {};
    hiprtc_validate(
      hiprtcGetCodeSize(
        kernel.program,
        &kernel_binary_size
      ),
      "failed to get program code size"
    );

    binary_list.resize(kernel_binary_size);
    hiprtc_validate(
      hiprtcGetCode(
        kernel.program,
        binary_list.data()
      ),
      "failed to get program code"
    );

    hiprtc_validate(
      hiprtcDestroyProgram(&kernel.program),
      "failed to destroy program"
    );

    hip_validate(
      hipModuleLoadData(&kernel.module, binary_list.data()),
      "failed to load module data"
    );

    for (bicudo::gpu::rocm_function &function : kernel.function_list) {
      hip_validate(
        hipModuleGetFunction(
          &function.entry_point,
          kernel.module,
          function.p_entry_point
        ),
        "failed to get entry point from module"
      );

      for (bicudo::gpu::buffer_t &buffer : function.buffer_list) {
        hip_validate(
          hipMalloc(
            &buffer.p_device,
            buffer.size
          ),
          "failed to allocate a-memory"
        );

        hip_validate(
          hipMemcpy(
            buffer.p_device,
            buffer.p_host,
            buffer.size,
            hipMemcpyHostToDevice
          ),
          "failed to copy memory"
        );

        function.argument_list.push_back(buffer.p_device);
        function.mem_size += buffer.size;
      }
    }
  }

  return result;
}