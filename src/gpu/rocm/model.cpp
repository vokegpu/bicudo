#include <bicudo/gpu/rocm/model.hpp>

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

bicudo::result_t bicudo::rocm::gpu_create_pipeline(
  bicudo::gpu_rm_divine_pipeline_t &pipeline
) {
  bicudo::log(bicudo::logtp(pipeline), "Creating...");

  std::vector<char> kernel_binary {};
  for (bicudo::gpu_rm_divine_kernel_t &kernel : pipeline.kernels) {
    if (kernel.status == bicudo::result::KERNEL_NOT_LOADED) {
      bicudo::logw(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Skipping... kernel not loaded.");
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
      hipModuleLoad(
        &kernel.hip_module,
        kernel_binary.data()
      ),
      hipSuccess,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Failed to load module.")
    );

    bicudo::log(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Module was loaded.");

    for (bicudo::gpu_rm_divine_fun_t &fun : kernel.functions) {
      bicudo::log(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Fetching entry-point '", fun.entry_point.name, "'...");

      bicudo_hip_assert(
        hipModuleGetFunction(
          &fun.entry_point.h_fun,
          kernel.hip_module,
          fun.entry_point.name.c_str()
        ),
        hipSuccess,
        bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Failed to fetch entry-point '", fun.entry_point.name, "'.")
      );
    }
  }

  return bicudo::result::SUCCESS;
}

bicudo::result_t bicudo::rocm::gpu_load_kernels(
  bicudo::gpu_rm_divine_pipeline_t &pipeline,
  bicudo::gpu_rm_divine_kernels_t &kernels
) {
  bicudo::log("Loading the ", kernels.size(), " kernels to pipeline '", pipeline.tag, "'...");
  
  std::string compile_program_log {};
  for (bicudo::gpu_rm_divine_kernel_t &kernel : kernels) {
    bicudo::log("Compiling kernel '" , kernel.tag, "'...");

    kernel.status = bicudo::result::KERNEL_NOT_LOADED;

    bicudo_hip_assert(
      hiprtcCreateProgram(
        &kernel.hip_program,
        kernel.src.c_str(),
        kernel.tag.c_str(),
        0, nullptr, nullptr
      ),
      HIPRTC_SUCCESS,
      bicudo::loge("Could not create program - may imortal Spirit of man failed.")
    );

    bicudo_hip_assert(
      hiprtcCompileProgram(
        kernel.hip_program,
        0, nullptr
      ),
      HIPRTC_SUCCESS,
      bicudo::loge("Could not compile kernel source, program has failed to emerge.")
    );

    std::size_t logsize {};
    bicudo_hip_assert(
      hiprtcGetProgramLogSize(
        kernel.hip_program,
        &logsize
      ),
      HIPRTC_SUCCESS,
      bicudo::loge("Could not get the length of the hypermachine error logs.")
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
        bicudo::loge("Could not get hypermachine logs to relate the failure of the kernel source Verb.")
      );

      bicudo::loge("Complete failure of Divine interaction.");
      continue;
    }

    bicudo::log("A program to interact with the kernel '", kernel.tag, "' source was created.");

    kernel.status = bicudo::result::KERNEL_LOADED;
    pipeline.kernels.emplace_back() = kernel;
  }

  return bicudo::result::SUCCESS;
}
