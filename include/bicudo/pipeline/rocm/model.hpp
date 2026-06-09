#ifndef BICUDO_PIPELINE_ROCM_MODEL_HPP
#define BICUDO_PIPELINE_ROCM_MODEL_HPP

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <bicudo/pipeline/base.hpp>
#include <hip/hip_runtime.h>

namespace bicudo::pipeline {
  class rocm : public bicudo::pipeline::base {
  protected:

  public:
    bicudo::result_t init() override;
  };
}

#endif
