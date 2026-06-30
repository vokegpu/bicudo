#include <bicudo/gpu/rocm/model.hpp>

bicudo::gpu_rm_divine_pipeline_t &bicudo::rocm::gpu_pipeline_new() {
  return *(this->pipelines.emplace_back() = new bicudo::gpu_rm_divine_pipeline_t {});
}

bicudo::result_t bicudo::rocm::init() {
  bicudo::device_id_t device_count {};
  if (hipGetDeviceCount(&device_count) != hipSuccess) {
    bicudo::loge("Failed to initialize ROCm pipeline; no device found");
    return bicudo::result::FAILED_TO_INITIALIZE_ROCM;
  }

  bicudo::log("HIP runtime found ", device_count, " devices!");

  for (bicudo::device_id_t i {}; i < device_count; i++) {
    hipDeviceProp_t deviceProp {};
    if (hipGetDeviceProperties(&deviceProp, i) != hipSuccess) {
      bicudo::logw("HIP runtime could not get device properties; device ", i, " !");
      continue;
    }

    bicudo::log("Device name: ", deviceProp.name);
  }

  if (hipSetDevice(pipeline_config.set_order) != hipSuccess) {
    bicudo::log("HIP runtime could not set interal device; check device set order to be zero or an existent device id!");
    return bicudo::result::FAILED_TO_INITIALIZE_ROCM;
  }

  bicudo::log("Bicudo selected device ID ", pipeline_config.set_order);

  return bicudo::result::SUCCESS;
}

bicudo::result_t bicudo::rocm::gpu_pipeline_create(
  bicudo::gpu_rm_divine_pipeline_t &pipeline
) {
  bicudo::log(bicudo::logtp(pipeline), "Creating...");

  bicudo::result_t status {bicudo::result::SUCCESS};
  std::vector<char> kernel_binary {};

  for (bicudo::gpu_rm_divine_kernel_t &kernel : pipeline.kernels) {
    if (kernel.status == bicudo::result::KERNEL_NOT_LOADED) {
      bicudo::logw(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Skipping... kernel not loaded.");
      status = bicudo::result::OK;
      continue;
    }

    bicudo::log(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Fetching kernel...");

    std::size_t kernel_binary_size {};
    bicudo_hip_assert(
      hiprtcGetCodeSize(
        kernel.hip_program,
        &kernel_binary_size
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Could not get compiled program binary length.")
    );

    kernel_binary.resize(kernel_binary_size);
    bicudo_hip_assert(
      hiprtcGetCode(
        kernel.hip_program,
        kernel_binary.data()
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Could not get compiled program binary.")
    );

    bicudo_hip_assert(
      hiprtcDestroyProgram(
        &kernel.hip_program
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Could not destroy program from memory!")
    );

    bicudo_hip_assert(
      hipModuleLoadData(
        &kernel.hip_module,
        kernel_binary.data()
      ),
      hipSuccess,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Failed to load module.")
    );

    bicudo::log(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Module was loaded.");

    for (bicudo::gpu_rm_divine_fun_t &fun : kernel.functions) {
      fun.unique_id = this->infspirit++;

      hipError_t result {};
      bicudo_hip_assert(
        (
          result =
            hipModuleGetFunction(
            &fun.entry_point.h_fun,
            kernel.hip_module,
            fun.entry_point.name.c_str()
          )
        ),
        hipSuccess,
        bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Failed to fetch entry-point '", fun.entry_point.name, "'.")
      );

      if (result != hipSuccess) {
        status = bicudo::result::OK;
        continue;
      }

      bicudo::log(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Entry-point '", fun.entry_point.name, "' was fetched.");

      for (bicudo::gpu_rm_divine_fun_args_t &arg : fun.args) {
        //bicudo_hip_assert(
        //  hipHostAlloc(
        //    &arg.p_pined_host,
        //    arg.bytes,
        //    0
        //  ),
        //  hipSuccess,
        //  bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Failed to malloc DEVICE memory of argument '", arg.tag,"'")
        //);

      //  hipHostMalloc(
        //  &arg.p_pinned_host,
        //  arg.bytes,
        //  0
        //);
//
  //      auto r = hipHostGetDevicePointer(
        //  &arg.p_pinned_device,
        //  arg.p_pinned_host,
        //  0
        //);
//
        //bicudo_trace_log(r == hipErrorOutOfMemory);

        //hipMemcpy(
        //  arg.p_pined_host,
        //  arg.p_host,
        //  arg.bytes,
        //  hipMemcpyHostToHost
        //);

        //hipMemcpy(
        //  arg.p_pined_device,
        //  arg.p_host,
        //  arg.bytes,
        //  hipMemcpyHostToDevice
        //);

        //bicudo_hip_assert(
        //  hipMemcpy(
        //    arg.p_pined_host,
        //    arg.p_host,
        //    arg.bytes,
        //    hipMemcpyHostToHost
        //  ),
        //  hipSuccess,
        //  bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Failed to memory copy from HOST to DEVICE of argument '", arg.tag,"'")
        //);

        bicudo::log(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Argument '", arg.tag, "' is atomic sacred now.");

        fun.memory.sacred_pointers.push_back(arg.p_device);
        fun.memory.sacred_pointers_mem_bytes_length += arg.bytes;
      }
    }
  }

  switch (status) {
  case bicudo::result::SUCCESS:
    bicudo::log(bicudo::logtp(pipeline), "Pipeline is created.");
    break;
  case bicudo::result::OK:
    bicudo::logw(bicudo::logtp(pipeline), "Some modules has failed to be instanced in pipeline creation - but pipeline is created.");
    break;
  default:
    break;
  }

  return bicudo::result::SUCCESS;
}

bicudo::result_t bicudo::rocm::gpu_pipeline_load_kernels(
  bicudo::gpu_rm_divine_pipeline_t &pipeline
) {
  bicudo::log(bicudo::logtp(pipeline), "Loading ", pipeline.kernels.size(), " kernels...");

  std::string compile_program_log {};
  for (bicudo::gpu_rm_divine_kernel_t &kernel : pipeline.kernels) {
    if (kernel.status == bicudo::result::KERNEL_NOT_LOADED) {
      continue;
    }

    bicudo::log(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Creating program...");

    kernel.unique_id = this->infspirit++;
    kernel.status = bicudo::result::KERNEL_NOT_LOADED;

    bicudo_hip_assert(
      hiprtcCreateProgram(
        &kernel.hip_program,
        kernel.src.c_str(),
        kernel.tag.c_str(),
        0, nullptr, nullptr
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Not loaded - Could not create program");
    );

    bicudo_hip_assert(
      hiprtcCompileProgram(
        kernel.hip_program,
        0, nullptr
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Not loaded - Could not compile program.")
    );

    std::size_t logsize {};
    bicudo_hip_assert(
      hiprtcGetProgramLogSize(
        kernel.hip_program,
        &logsize
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Not loaded - Could not get program log length.")
    );

    if (logsize) {
      kernel.status = bicudo::result::FAILED_TO_COMPILE_KERNEL;
      compile_program_log.resize(logsize);
     
      bicudo_hip_assert(
        hiprtcGetProgramLog(
          kernel.hip_program,
          compile_program_log.data()
        ),
        HIPRTC_SUCCESS,
        bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Not loaded - Could not get program log.")
      );

      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Not loaded - Failed to compile program.");
      bicudo::loge(compile_program_log);

      pipeline.kernels.emplace_back() = kernel;
      continue;
    }

    kernel.status = bicudo::result::KERNEL_LOADED;
    bicudo::log(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Loaded - program was created and sucessfully compiled.");
  }

  return bicudo::result::SUCCESS;
}
