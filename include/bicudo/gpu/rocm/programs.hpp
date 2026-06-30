#ifndef BICUDO_GPU_ROCM_PROGRAMS_HPP
#define BICUDO_GPU_ROCM_PROGRAMS_HPP

namespace bicudo::gpu::rocm {
  constexpr char* hip_rocm_kernel_runtime_assert {
    R"(

/**
 * This hip runtime should perform memory-access assert
 * to ROCm runtime initialization.
 **/

extern "C" __global__
void runtime_assert_entry_point(
  float *__restrict__ p_assert_buffer
) {
  p_assert_buffer[0] = 17.0f;
  p_assert_buffer[1] = 27.0f;
  p_assert_buffer[2] = 37.0f;
  p_assert_buffer[3] = 47.0f;
  p_assert_buffer[4] = 52.0f;
}

    )"
  };
}

#endif
