#ifndef BICUDO_PIPELINE_ROCM_MODEL_HPP
#define BICUDO_PIPELINE_ROCM_MODEL_HPP

#ifndef __HIP_PLATFORM_AMD__
#define __HIP_PLATFORM_AMD__
#endif

#include <bicudo/pipeline/base.hpp>
#include <hip/hip_runtime.h>

namespace bicudo {
  struct rocm_pipeline_configuration_t {
  public:
    bicudo::device_id_t set_order = bicudo::pipeline_device_set_order::FIRST_ONE;
  };
}

namespace bicudo::pipeline {
  class rocm : public bicudo::pipeline::base {
  protected:
    bicudo::rocm_pipeline_configuration_t pipeline_config;
  public:
    rocm(bicudo::rocm_pipeline_configuration_t pipeline_config) : base() {
      this->pipeline_config = pipeline_config;
    }
  public:
    bicudo::result_t init() override;
  };
}

#endif
