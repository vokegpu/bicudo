#include "bicudo/bicudo.hpp"
#include "bicudo/gpu/rocm.hpp"

#include <vector>
#include <iostream>

bicudo::application bicudo::app {};

void bicudo::init() {
  bicudo::app.world_manager.on_create();
}

void bicudo::update() {
  bicudo::app.world_manager.on_update();
}

void bicudo::render() {
  bicudo::app.world_manager.on_render();
}

void bicudo::viewport(int32_t w, int32_t h) {
  bicudo::app.world_manager.immediate.set_viewport(w, h);
}

void bicudo::world::insert(bicudo::object *p_obj) {
  bicudo::app.world_manager.push_back_object(p_obj);
}

bicudo::collided bicudo::world::pick(bicudo::object *&p_obj, const bicudo::vec2 &pos) {
  bicudo::vec4 rect {};
  for (bicudo::object *&p_objs : bicudo::app.world_manager.loaded_object_list) {
    rect.x = p_objs->placement.pos.x;
    rect.y = p_objs->placement.pos.y;
    rect.z = p_objs->placement.size.x;
    rect.w = p_objs->placement.size.y;

    if (bicudo::vec4_collide_with_vec2(rect, pos)) {
      p_obj = p_objs;
      return true;
    }
  }

  return false;
}

/**
#define assert_log(result, expect, error) result != expect && std::cout << error << std::endl

static constexpr auto kernel_source {
R"(
  extern "C"
  __global__ void meow(uint32_t *p_number) {
    if (threadIdx.x == 0) {
      uint32_t &number {*p_number};
      number += 250;
    }
  }
)"
};

void bicudo::count(uint32_t *p_number) {
  hiprtcProgram program {};

  hiprtcCreateProgram(
    &program,
    kernel_source,
    "meow.cpp",
    0,
    nullptr,
    nullptr
  );

  hiprtcCompileProgram(
    program,
    0,
    nullptr
  );

  uint64_t log_size {};
  hiprtcGetProgramLogSize(program, &log_size);

  if (log_size) {
    std::string log {}; log.resize(log_size);
    hiprtcGetProgramLog(program, log.data());
  
    std::cout << "meow??? " << log_size << std::endl;
  }

  hipDeviceProp_t device_properties {};
  assert_log(hipGetDeviceProperties(&device_properties, 0), hipSuccess, "idc");
  std::cout << device_properties.gcnArchName << std::endl;

  uint64_t kernel_binary_size {};
  assert_log(hiprtcGetCodeSize(program, &kernel_binary_size), HIP_SUCCESS, "meow");

  std::vector<char> kernel_binary(kernel_binary_size);
  assert_log(hiprtcGetCode(program, kernel_binary.data()), HIP_SUCCESS, "ugabuga");

  assert_log(hiprtcDestroyProgram(&program), HIPRTC_SUCCESS, "? failed to do meow");

  hipModule_t module {};
  hipFunction_t kernel {};

  assert_log(hipModuleLoadData(&module, kernel_binary.data()), hipSuccess, "idk");
  assert_log(hipModuleGetFunction(&kernel, module, "meow"), hipSuccess, "idk");

  uint32_t *p_number_device {};

  assert_log(hipMalloc(&p_number_device, sizeof(uint32_t)), hipSuccess, "oi");
  assert_log(hipMemcpy(p_number_device, &p_number, sizeof(uint32_t), hipMemcpyHostToDevice), hipSuccess, "oi");

  struct {
    uint32_t *p_number;
  } args {p_number_device};

  uint64_t size {sizeof(args)};

  void *p_configs[] {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, &args,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
    HIP_LAUNCH_PARAM_END
  };

  hipError_t err {
    hipModuleLaunchKernel(
      kernel,
      1,
      1,
      1,
      1,
      1,
      1,
      0,
      nullptr,
      nullptr,
      p_configs
    )
  };

  if (err != hipSuccess) {
    std::cout << "muu: " << hipGetErrorName(err) << std::endl;
  }

  assert_log(hipMemcpy(&p_number, p_number_device, sizeof(uint32_t), hipMemcpyDeviceToHost), hipSuccess, "oi amo brigadeiro");

  std::cout << "bicudo muuuuuuuuuuuu" << p_number << std::endl;
}
**/