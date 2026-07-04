#ifndef BICUDO_GPU_ROCM_PROGRAMS_HPP
#define BICUDO_GPU_ROCM_PROGRAMS_HPP

namespace bicudo::gpu::rocm {
  const char* kernel_collision_detection {
    R"(
      /**
       * This hip runtime should perform collision detection.
       **/

      extern "C" __global__
      void collision_detection_entry_point(
        float *__restrict__ *p_in_body_region,
        float *__restrict__ *p_out_body_region
      ) {
        
      }
    )"
  };
}

#endif

//
