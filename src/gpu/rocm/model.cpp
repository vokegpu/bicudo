#include <bicudo/gpu/rocm/model.hpp>
#include <bicudo/pipeline/rocm.hpp>
#include <bicudo/gpu/rocm/programs.hpp>
#include <bicudo/gpu/gpu.hpp>
#include <bicudo/gpu/rocm/divine.hpp>
#include <bicudo/core/core.hpp>

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
  /* GPU pick */

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

  /* assert */

  bicudo::result_t assert_testing_result {bicudo::result::OK};
  bicudo::log("Starting HIP ROCm assertation tests...");

  bicudo::rocm &rocm = bicudo::as_gpu<bicudo::rocm>();
  bicudo::gpu_rm_divine_pipeline_t &pipeline52 = bicudo::as_new_pipeline();

  pipeline52 = {
    .tag = "52", .description = "The divine kernel for Divine numbers assertation."
  };

  bicudo::gpu_sacred_atomic_memory_t atomic {.bytes = sizeof(float)*5};
  bicudo::gpu_sacred_allocate_atomic(atomic, hipHostMallocMapped, 0);

  bicudo::gpu_rm_divine_kernel_t &kernel_hip_runtime = bicudo::as_kernel(pipeline52);

  kernel_hip_runtime = {
    .tag = "hip-runtime-assert",
    .src = bicudo::hip_rocm_kernel_runtime_assert
  };

  kernel_hip_runtime.functions = {
    {
      .entry_point = {
        .name = "gpu_divine_main"
      }
    }
  };

  rocm.gpu_pipeline_load_kernels(pipeline52);
  rocm.gpu_pipeline_create(pipeline52);

  bicudo::gpu_rm_divine_module_t &module_hip_runtime_assert {
    bicudo::as_module(pipeline52, "hip-runtime-assert")
  };

  bicudo::gpu_rm_divine_fun_t &fun_entry_point_hip_runtime_assert {
    bicudo::as_function(module_hip_runtime_assert, "gpu_divine_main")
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

    module_hip_runtime_assert.dispatch = {
      .dimension = {
        .grid = bicudo::vec3_t<uint32_t>(1, 1, 1),
        .block = bicudo::vec3_t<uint32_t>(1, 1, 1)
      }
    };

    bicudo::gpu_sacred_arguments_t arguments {
      {
        .tag = "p_assert_buffer",
        .p_device = atomic.p_device,
        .bytes = atomic.bytes
      }
    };


    bicudo::gpu_sacred_dispatch_params_t params {};
    bicudo::gpu_sacred_param_pointers_t points {};

    bicudo::gpu_sacred_fetch_dispatch_params(
      module_hip_runtime_assert.dispatch,
      params,
      arguments,
      points
    );

    bicudo::gpu_sacred_call(
      module_hip_runtime_assert,
      fun_entry_point_hip_runtime_assert,
      module_hip_runtime_assert.dispatch
    );

    bicudo::gpu_sacred_async_fetch(
      module_hip_runtime_assert.dispatch.memory,
      atomic.p_host,
      atomic.p_device,
      atomic.bytes
    );

    for (std::size_t i = 0; i < 5; i++) {
      bicudo::log("Checking assertations -> ", p[i]);
    }

    bicudo_assert(
      p[0], 17.0f,
      bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[0], " but MUST be 17.0f.")
    );

    bicudo_assert(
      p[1], 27.0f,
      bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[1], " but MUST be 27.0f.")
    );

    bicudo_assert(
      p[2], 37.0f,
      bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[2], " but MUST be 37.0f.")
    );

    bicudo_assert(
      p[3], 47.0f,
      bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[3], " but MUST be 47.0f.")
    );

    bicudo_assert(
      p[4], 52.0f,
      bicudo::loge("Potentially issue with your GPU be careful, receveid ", p[4], " but MUST be 52.0f.")
    );

    break;
  }

  bicudo::gpu_free_sacred_atomic(atomic);
  rocm.gpu_pipeline_free(pipeline52);

  /* detection */

  bicudo::log("Initializing the detection pipeline.");

  this->pipeline_collision_detection.unique_id = this->infspirit++;
  this->pipelines.push_back(&this->pipeline_collision_detection);

  bicudo::gpu_rm_divine_kernel_t &collision_kernel = bicudo::as_kernel(this->pipeline_collision_detection);

  collision_kernel.tag = "collision-detection";
  collision_kernel.src = bicudo::hip_rocm_kernel_collision_detection;

  collision_kernel.functions = {
    {
      .entry_point = {
        .name = "gpu_divine_main"
      }
    }
  };

  rocm.gpu_pipeline_load_kernels(this->pipeline_collision_detection);
  rocm.gpu_pipeline_create(this->pipeline_collision_detection);

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

bicudo::result_t bicudo::rocm::registry_hypergroup(bicudo::hypergroup_t *p_hypergroup) {
  p_hypergroup->unique_id = this->infspirit++;
  this->hypergroups.push_back(p_hypergroup);
  return bicudo::result::OK;
}

bicudo::result_t bicudo::rocm::registry_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body
) {
  p_body->unique_id = this->infspirit++;
  p_hypergroup->bodies.push_back(p_body);
  return bicudo::result::OK;
}

bicudo::result_t bicudo::rocm::unregistry_hypergroup(
  bicudo::hypergroup_t *p_hypergroup
) {
  for (std::size_t i {}; i < this->hypergroups.size(); i++) {
    if (this->hypergroups.at(i) != p_hypergroup) continue;
    this->hypergroups.erase(this->hypergroups.begin() + i);
    return bicudo::result::SUCCESS;
  }

  return bicudo::result::FAILED;
}

bicudo::result_t bicudo::rocm::unregistry_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body) {

  for (bicudo::hypergroup_t *p_hp : this->hypergroups) {
    if (p_hp != p_hypergroup) continue;
    for (std::size_t i {}; i < p_hp->bodies.size(); i++) {
      if (p_hp->bodies.at(i) != p_body) continue;
      p_hp->bodies.erase(p_hp->bodies.begin() + i);
      return bicudo::result::SUCCESS;
    }
  }

  return bicudo::result::FAILED;
}

void bicudo::rocm::update_hypergroup_sacred_atomic_machine(
  bicudo::hypergroup_t &hypergroup
) {
  switch (hypergroup.state) {
    case bicudo::hypergroup_state::UNHANDLED: {
      hypergroup.state = bicudo::hypergroup_state::MUST_REFRESH_HOST;
      break;
    }
    
    case bicudo::hypergroup_state::MUST_REFRESH_HOST: {
      std::size_t enough_bodies_bytes {
        hypergroup.bodies.size()
        *
        sizeof(bicudo::body_rect_memory_layout_t)
        *
        this->vram_fraction_usage
      };

      bicudo::log(bicudo::logt(hypergroup, "hypergroup"), "Refreshing HOST for memory region...");
      
      switch (bicudo::gpu_sacred_reallocate_atomic(hypergroup.atomic, enough_bodies_bytes, 0, 0)) {
        case bicudo::result::SUCCESS: {
          hypergroup.state = bicudo::hypergroup_state::MUST_DISPATCH;
          hypergroup.has_memory_filled_once = false;

          bicudo::gpu_sacred_arguments_t arguments {
            {
              .tag = "p_hd_hypergroup_bodies",
              .p_device = hypergroup.atomic.p_device,
              .bytes = hypergroup.atomic.bytes
            }
          };

          bicudo::gpu_rm_divine_module_t &km {
            bicudo::as_module(this->pipeline_collision_detection, "collision-detection")
          };

          bicudo::gpu_sacred_fetch_dispatch_params(
            km.dispatch,
            hypergroup.ref_params,
            arguments,
            hypergroup.raw_params
          );

          bicudo::log(bicudo::logt(hypergroup, "hypergroup"), "HOST (", (enough_bodies_bytes), " bytes) memory region is safety now");
          break;
        }
        default: {
          bicudo::loge(bicudo::logt(hypergroup, "hypergroup"), "Could not refresh HOST for memory region!");
          break;
        }
      }
  
      break;
    }

    case bicudo::hypergroup_state::MUST_DISPATCH: {
      if (!hypergroup.has_memory_filled_once) {
        bicudo::log(bicudo::logt(hypergroup, "hypergroup"), "Waiting for memory be filled once.");
        break;
      }

      bicudo::log(bicudo::logt(hypergroup, "hypergroup"), "Dispatching detection module...");

      bicudo::gpu_rm_divine_module_t &km {
        bicudo::as_module(this->pipeline_collision_detection, "collision-detection")
      };

      bicudo::gpu_rm_divine_fun_t &fun {
        bicudo::as_function(km, "gpu_divine_main")
      };

      km.dispatch.memory.params = hypergroup.ref_params;
      km.dispatch.dimension.grid = bicudo::vec3_t<uint32_t>(1, 1, 1);
      km.dispatch.dimension.block = bicudo::vec3_t<uint32_t>(hypergroup.bodies_pass_count, hypergroup.bodies_pass_count, 1);

      bicudo::gpu_sacred_call(
        km,
        fun,
        km.dispatch
      );

      hypergroup.elapsed = std::chrono::steady_clock::now();
      hypergroup.state = bicudo::hypergroup_state::MUST_WAIT;

      break;
    }

    case bicudo::hypergroup_state::MUST_WAIT: {
      std::chrono::steady_clock::time_point now = std::chrono::steady_clock::now();
      if (
        (
          std::chrono::duration_cast<std::chrono::milliseconds>(hypergroup.elapsed - now).count()
        )
        <
        this->gpu_sync
      ) {
        break;
      }

      bicudo::log("Syncing memory...");

      bicudo::gpu_rm_divine_module_t &km {
        bicudo::as_module(this->pipeline_collision_detection, "collision-detection")
      };

      bicudo::gpu_sacred_async_fetch(
        km.dispatch.memory,
        hypergroup.atomic.p_host,
        hypergroup.atomic.p_device,
        hypergroup.atomic.bytes
      );

      break;
    }

    case bicudo::hypergroup_state::MUST_REFILL: {
      if (!hypergroup.has_memory_filled_once) {
        hypergroup.state = bicudo::hypergroup_state::MUST_DISPATCH;
      }

      break;
    }

    case bicudo::hypergroup_state::IDLE: {
      if (!hypergroup.has_memory_filled_once) {
        break;
      }

      bicudo::logw("... -> ", hypergroup.bodies_pass_count);

      bicudo::gpu_rm_divine_module_t &km {
        bicudo::as_module(this->pipeline_collision_detection, "collision-detection")
      };

      bicudo::gpu_sacred_async_fetch(
        km.dispatch.memory,
        hypergroup.atomic.p_host,
        hypergroup.atomic.p_device,
        hypergroup.atomic.bytes
      );

      float *p = static_cast<float*>(hypergroup.atomic.p_host);
      hypergroup.has_memory_filled_once = false;

      for (std::size_t i {}; i < hypergroup.bodies_pass_count; i++) {
        bicudo::logw("-> [", i, "] -> ", p[i]);
      }

      bicudo::logw("IDLE mode...");

      break;
    }
  }
}

bicudo::result_t bicudo::rocm::update_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body
) {
  if (p_hypergroup->state == bicudo::hypergroup_state::MUST_REFRESH_HOST) {
    return bicudo::result::OK;
  }

  p_body->velocity += p_body->acceleration * bicudo::dt;
  p_body->pos += p_body->velocity;

  p_body->angular_velocity += p_body->angle_acceleration * bicudo::dt;
  p_body->angle += p_body->angular_velocity;

  float midw {p_body->size.x / 2};
  float midh {p_body->size.y / 2};

  switch (p_hypergroup->state) {
    case bicudo::hypergroup_state::MUST_DISPATCH: {
      if (p_body->vertices.empty()) {
        p_body->vertices.resize(4);
      }

      p_body->vertices.at(0) = bicudo::vec2_t<float>(p_body->pos.x - midw, p_body->pos.y - midh);
      p_body->vertices.at(1) = bicudo::vec2_t<float>(p_body->pos.x + midw, p_body->pos.y - midh);
      p_body->vertices.at(2) = bicudo::vec2_t<float>(p_body->pos.x + midw, p_body->pos.y + midh);
      p_body->vertices.at(3) = bicudo::vec2_t<float>(p_body->pos.x - midw, p_body->pos.y + midh);
    
      for (bicudo::vec2_t<float> &vertex : p_body->vertices) {
        vertex = vertex.rotate(p_body->angular_velocity, p_body->pos);
    
        p_body->min.x = std::min(p_body->min.x, vertex.x);
        p_body->min.y = std::min(p_body->min.y, vertex.y);
        p_body->max.x = std::max(p_body->max.x, vertex.x);
        p_body->max.y = std::max(p_body->max.y, vertex.y);

        if (!this->is_sacred_context) continue;

        /**
         * We pass the vertex content to the HOST
         * atomic before dispatching to GPU.
         **/
    
        bicudo::pass_memory<float>(
          p_hypergroup->atomic.p_host,
          p_hypergroup->body_byte_index,
          vertex
        );

        p_hypergroup->has_memory_filled_once = true;
      }

      /**
       * Has collide flag.
       **/

      bicudo::pass_memory<float>(
        p_hypergroup->atomic.p_host,
        p_hypergroup->body_byte_index,
        {0.0f, 0.0f}
      );

      break;
    }

    case bicudo::hypergroup_state::MUST_REFILL: {
      p_hypergroup->body_byte_index += bicudo::body_rect_vertexes_unit;
      float *p = static_cast<float*>(p_hypergroup->atomic.p_host);

      float has_collided = p[p_hypergroup->body_byte_index];
      if (has_collided != 0.0f) {
        p_body->has_collide = true;
      } else {
        p_body->has_collide = false;
      }

      p_hypergroup->has_memory_filled_once = false;
      break;
    } 

    default: {
      break;
    }
  }

  p_body->rect.x = p_body->pos.x - midw;
  p_body->rect.y = p_body->pos.y - midh;
  p_body->rect.z = p_body->size.x;
  p_body->rect.w = p_body->size.y;

  p_hypergroup->bodies_pass_count += bicudo::body_rect_vertexes_unit;
  return bicudo::result::OK;
}

bicudo::result_t bicudo::rocm::update(
  bicudo::physics_update_mode mode
) {
  this->is_sacred_context = true;

  if (mode != bicudo::physics_update_mode::EVERYTHING) {
    bicudo::logw("For GPU-acceleration, physics mode only support EVERYTHING");
  }

  std::size_t it_per_collide_solve {15};
  float correction_rate {0.8f};
  float bytes_per_all_bodies_vertices {};

  for (bicudo::hypergroup_t *p_hypergroup : this->hypergroups) {
    this->update_hypergroup_sacred_atomic_machine(
      *p_hypergroup
    );

    p_hypergroup->bodies_pass_count = 0;
    p_hypergroup->body_byte_index = 0;

    for (bicudo::body_t *p_body : p_hypergroup->bodies) {
      this->update_body(
        p_hypergroup,
        p_body
      );
    }
  }

  this->is_sacred_context = false;
  return bicudo::result::OK;
}

bicudo::result_t bicudo::rocm::size_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body,
  bicudo::vec2_t<float> size
) {
  return bicudo::result::OK;
}

bicudo::result_t bicudo::rocm::move_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body,
  bicudo::vec2_t<float> direction
) {
  return bicudo::result::OK;
}
