#ifndef BICUDO_PIPELINE_ROCM_MODEL_HPP
#define BICUDO_PIPELINE_ROCM_MODEL_HPP

#include <bicudo/pipeline/base.hpp>
#include <bicudo/gpu/rocm/divine.hpp>
#include <bicudo/gpu/rocm/sacred.hpp>
#include <bicudo/gpu/rocm/header.hpp>
#include <vector>

namespace bicudo {
  struct rocm_pipeline_configuration_t {
  public:
    bicudo::device_id_t set_order = bicudo::pipeline_device_set_order::FIRST_ONE;
  };
}

namespace bicudo {
  class rocm : public bicudo::pipeline::base {
  protected:
    bicudo::rocm_pipeline_configuration_t pipeline_config {};
    bicudo::id_t infspirit {};
    std::vector<bicudo::gpu_rm_divine_pipeline_t*> pipelines {};
  public:
    rocm(bicudo::rocm_pipeline_configuration_t pipeline_config) : base() {
      this->pipeline_config = pipeline_config;
    }
  public:
    bicudo::result_t init() override;
  public:
    bicudo::gpu_rm_divine_pipeline_t &gpu_pipeline_new();

    bicudo::result_t gpu_pipeline_create(
      bicudo::gpu_rm_divine_pipeline_t &pipeline
    );

    bicudo::result_t gpu_pipeline_load_kernels(
      bicudo::gpu_rm_divine_pipeline_t &pipeline
    );

    bicudo::result_t gpu_pipeline_free(
      bicudo::gpu_rm_divine_pipeline_t &pipeline
    );
  };
}

#endif
