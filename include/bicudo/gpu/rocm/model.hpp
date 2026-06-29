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
    bicudo::rocm_pipeline_configuration_t pipeline_config;
  public:
    rocm(bicudo::rocm_pipeline_configuration_t pipeline_config) : base() {
      this->pipeline_config = pipeline_config;
    }
  public:
    bicudo::result_t init() override;
  public:
    bicudo::result_t gpu_pipeline_create(
      bicudo::gpu_rm_divine_pipeline_t &pipeline
    );

    bicudo::result_t gpu_pipeline_load_kernels(
      bicudo::gpu_rm_divine_pipeline_t &pipeline
    );

    bicudo::result gpu_pipeline_get_module_by_index(
      bicudo::gpu_rm_divine_pipeline_t &pipeline,
      bicudo::gpu_rm_divine_module_t &kmodule,
      std::size_t index
    );

    bicudo::result gpu_pipeline_get_module_by_tag(
      bicudo::gpu_rm_divine_pipeline_t &pipeline,
      bicudo::gpu_rm_divine_module_t &kmodule,
      const std::string &tag
    );

    bicudo::result gpu_pipeline_get_function_by_index(
      bicudo::gpu_rm_divine_pipeline_t &pipeline,
      bicudo::gpu_rm_divine_module_t &kmodule,
      bicudo::gpu_rm_divine_fun_t &fun,
      std::size_t index
    );

    bicudo::result gpu_pipeline_get_function_by_name(
      bicudo::gpu_rm_divine_pipeline_t &pipeline,
      bicudo::gpu_rm_divine_module_t &kmodule,
      bicudo::gpu_rm_divine_fun_t &fun,
      const std::string &name
    );

    bicudo::result_t gpu_pipeline_invoke_function(
      bicudo::gpu_rm_divine_pipeline_t &pipeline,
      bicudo::gpu_rm_divine_fun_t &fun
    );
  };
}

#endif
