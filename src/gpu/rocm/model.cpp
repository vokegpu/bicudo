#include <bicudo/gpu/rocm/model.hpp>
#include <bicudo/pipeline/rocm.hpp>
#include <bicudo/gpu/rocm/programs.hpp>
#include <bicudo/gpu/gpu.hpp>

bicudo::gpu_rm_divine_pipeline_t &bicudo::rocm::gpu_pipeline_new() {
  return *(this->pipelines.emplace_back() = new bicudo::gpu_rm_divine_pipeline_t { .unique_id = this->infspirit++ });
}

bicudo::result_t bicudo::rocm::gpu_pipeline_free(
  bicudo::gpu_rm_divine_pipeline_t &pipeline
) {
  for (std::size_t i {}; i < this->pipelines.size(); i++) {
    bicudo::gpu_rm_divine_pipeline_t &p = *this->pipelines.at(i);
    if (p == pipeline) {
      for (bicudo::gpu_rm_divine_module_t kmodule : p.kernels) {
        bicudo_assert(
          hipModuleUnload(
            kmodule.hip_module
          ),
          hipSuccess,
          bicudo::loge(bicudo::logtp(p), bicudo::logtk(kmodule), "Could not unload from divine memory.");
        );

        bicudo::log(bicudo::logtp(p), bicudo::logtk(kmodule), "Module was unload from divine memory.");
      }

      bicudo::log(bicudo::logtp(p), "Pipeline was free to the eternity.");
      this->pipelines.erase(this->pipelines.begin() + i);

      return bicudo::result::SUCCESS;
    }
  }

  return bicudo::result::PIPELINE_NOT_FOUND;
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

  /* testing purpose */

  bicudo::log("Starting HIP ROCm assertation tests...");

  bicudo::result_t assert_testing_result {bicudo::result::OK};

  char* hip_rocm_kernel_runtime_assert {
    R"(
      /**
       * This hip runtime should perform memory-access assert
       * to ROCm runtime initialization.
       **/
      
      extern "C" __global__
      void runtime_assert_entry_point(
        float *__restrict__ p_assert_buffer
      ) {
        p_assert_buffer[0] = 17.0f; // from 153.0f
        p_assert_buffer[1] = 27.0f; // from 26.0f
        p_assert_buffer[2] = 37.0f; // from 24.0f
        p_assert_buffer[3] = 47.0f; // from 6.0f
        p_assert_buffer[4] = 52.0f; // from 1977.0f
      }
    )"
  };

  bicudo::rocm &rocm = *this;
  bicudo::gpu_rm_divine_pipeline_t &pipeline52 = rocm.gpu_pipeline_new();

  pipeline52 = {
    .tag = "52", .description = "The divine kernel for Divine numbers assertation."
  };

  bicudo::gpu_rm_sacred_atomic_memory_t atomic(sizeof(float)*5);
  bicudo::gpu_allocate_sacred_atomic(atomic, hipHostMallocMapped, 0);

  bicudo::gpu_rm_divine_kernel_t &kernel_hip_runtime = bicudo::as_kernel(pipeline52);

  kernel_hip_runtime = {
    .tag = "hip-runtime-assert",
    .src = hip_rocm_kernel_runtime_assert
  };

  kernel_hip_runtime.functions = {
    {
      .entry_point = {
        .name = "runtime_assert_entry_point"
      },
      .memory = {
        .hip_stream = nullptr
      },
      .dimension = {
        .grid = bicudo::vec3_t<uint32_t>(1, 1, 1),
        .block = bicudo::vec3_t<uint32_t>(1, 1, 1)
      },
      .args = {
        {
          .tag = "assert-buf",
          .bytes = atomic.bytes,
          .p_device = atomic.p_device
        }
      }
    }
  };

  rocm.gpu_pipeline_load_kernels(pipeline52);
  rocm.gpu_pipeline_create(pipeline52);

  bicudo::gpu_rm_divine_module_t &module_hip_runtime_assert {
    bicudo::as_module(pipeline52, "hip-runtime-assert")
  };

  bicudo::gpu_rm_divine_fun_t &fun_entry_point_hip_runtime_assert {
    bicudo::as_function(module_hip_runtime_assert, "runtime_assert_entry_point")
  };

  if (module_hip_runtime_assert != bicudo::found || fun_entry_point_hip_runtime_assert != bicudo::found) {
    assert_testing_result = bicudo::result::FAILED;
    bicudo::loge("Unknown reason to: hip-runtime-module or hip-runtime-entry-point be not found");
  }

  switch (assert_testing_result) {
  case bicudo::result::FAILED:
    break;
  case bicudo::result::OK:
    float *p = static_cast<float*>(atomic.p_host);

    p[0] = 153.0f;  // should be 17.0f
    p[1] = 26.0f;   // should be 27.0f
    p[2] = 24.0f;   // should be 37.0f
    p[3] = 6.0f;    // should be 47.0f
    p[4] = 1977.0f; // should be 52.0f

    bicudo::gpu_sacred_call(
      module_hip_runtime_assert,
      fun_entry_point_hip_runtime_assert
    );

    bicudo::gpu_sacred_async_fetch(
      fun_entry_point_hip_runtime_assert,
      atomic.p_host,
      atomic.p_device,
      atomic.bytes
    );

    for (std::size_t i = 0; i < 5; i++) {
      bicudo::log("Checking assertations -> ", p[i]);
    }

    bicudo_assert(p[0], 17.0f, bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[0], " but MUST be 17.0f."));
    bicudo_assert(p[1], 27.0f, bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[1], " but MUST be 27.0f."));
    bicudo_assert(p[2], 37.0f, bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[2], " but MUST be 37.0f."));
    bicudo_assert(p[3], 47.0f, bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[3], " but MUST be 47.0f."));
    bicudo_assert(p[4], 52.0f, bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[4], " but MUST be 52.0f."));

    break;
  }

  bicudo::gpu_free_sacred_atomic(atomic);
  rocm.gpu_pipeline_free(pipeline52);

  return assert_testing_result;
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
    bicudo_assert(
      hiprtcGetCodeSize(
        kernel.hip_program,
        &kernel_binary_size
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Could not get compiled program binary length.")
    );

    kernel_binary.resize(kernel_binary_size);
    bicudo_assert(
      hiprtcGetCode(
        kernel.hip_program,
        kernel_binary.data()
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Could not get compiled program binary.")
    );

    bicudo_assert(
      hiprtcDestroyProgram(
        &kernel.hip_program
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Could not destroy program from memory!")
    );

    bicudo_assert(
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
      bicudo_assert(
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

    bicudo_assert(
      hiprtcCreateProgram(
        &kernel.hip_program,
        kernel.src.c_str(),
        kernel.tag.c_str(),
        0, nullptr, nullptr
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Not loaded - Could not create program");
    );

    bicudo_assert(
      hiprtcCompileProgram(
        kernel.hip_program,
        0, nullptr
      ),
      HIPRTC_SUCCESS,
      bicudo::loge(bicudo::logtp(pipeline), bicudo::logtk(kernel), "Not loaded - Could not compile program.")
    );

    std::size_t logsize {};
    bicudo_assert(
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
     
      bicudo_assert(
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
