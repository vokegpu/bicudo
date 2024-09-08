#include "model.hpp"
#include "util/log.hpp"

bicudo::result bicudo::dispatch(
  bicudo::gpu::pipeline *p_pipeline,
  uint64_t module_index,
  uint64_t kernel_index,
) {
  bicudo::gpu::kernel &kernel {p_pipeline->kernel_list.at(module_index)};
  bicudo::gpu::function &function {kernel.function_list.at(kernel_index)};

  void *p_configs[] {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, function.argument_list.data(),
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
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
      << hiprtcGetErrorString(err) << std::endl;
    return bicudo::types::FAILED;
  }

  return bicudo::types::SUCCESS;
}

bicudo::result bicudo::writeback(
  bicudo::gpu::pipeline *p_pipeline,
  uint64_t module_index,
  uint64_t kernel_index
) {
  bicudo::gpu::kernel &kernel {p_pipeline->kernel_list.at(module_index)};
  bicudo::gpu::function &function {kernel.function_list.at(kernel_index)};

  hip_validate(
    hipMemcpy(
      function.p_data,
      function.p_data,
      sizeof(function.size),
      hipMemcpyDeviceToHost
    ),
    "failed to copy memory"
  );

  return bicudo::types::SUCCESS;
}

bicudo::result bicudo::create_pipeline(
  bicudo::gpu::pipeline *p_pipeline,
  bicudo::gpu::pipeline_create_info *p_pipeline_create_info
) {
  p_pipeline->kernel_list = p_pipeline_create_info->kernel_list;
  bicudo::result result {bicudo::types::SUCCESS};
  std::vector<char> binary_list {};

  for (bicudo::gpu::kernel &kernel : p_pipeline->kernel_list) {
    hiprtc_validate(
      hiprtcCreateProgram(
        kernel.program,
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

      std::cout << "[GPU] failed to create program, more info:\n" << log << std::endl;
      result = bicudo::types::FAILED;
      continue;
    }

    uint64_t kernel_binary_size {};
    hiprtc_validate(
      hiprtcGetCodeSize(
        kernel.program,
        kernel_binary_size
      ),
      "failed to get program code size"
    );

    binary_list.resize(kernel_binary_size);
    hiprtc_validate(
      hiprtcGetCode(
        kernel.program,
        kernel_binary.data()
      ),
      "failed to get program code"
    );

    hiprtc_validate(
      hiprtcDestroyProgram(&kernel.program),
      "failed to destroy program"
    );

    hip_validate(
      hipModuleLoadData(kernel.module, binary_list.data()),
      "failed to load module data"
    );

    for (bicudo::gpu::function &function : kernel.function_list) {
      hip_validate(
        hipModuleGetFunction(
          &function.entry_point,
          kernel.module,
          function.p_entry_point
        ),
        "failed to get entry point from module"
      );

      for (bicudo::gpu::buffer &buffer : function.buffer_list) {
        hip_validate(
          hipMalloc(
            &function.p_data,
            buffer.size
          ),
          "failed to allocate a-memory"
        );

        hip_validate(
          hipMemcpy(
            function.p_data,
            function.p_data,
            sizeof(function.size),
            hipMemcpyHostToDevice
          ),
          "failed to copy memory"
        );

        function.argument_list.push_back(function.p_data);
        function.mem_size += buffer.size;
      }
    }
  }

  return result;
}