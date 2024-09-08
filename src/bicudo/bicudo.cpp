#include "bicudo/bicudo.hpp"

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

#include <vector>
#include <iostream>

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

  uint32_t p_number {};
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