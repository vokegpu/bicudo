#ifndef BICUDO_PIPELINE_ROCM_MODEL_HPP
#define BICUDO_PIPELINE_ROCM_MODEL_HPP

#include <bicudo/pipeline/base.hpp>
#include <bicudo/gpu/rocm/divine.hpp>
#include <bicudo/gpu/rocm/sacred.hpp>
#include <bicudo/gpu/rocm/header.hpp>
#include <bicudo/physics/hypergroup.hpp>
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
    std::vector<bicudo::gpu_rm_divine_pipeline_t*> pipelines {};
    std::vector<bicudo::hypergroup_t*> hypergroups {};
    bicudo::id_t infspirit {};
    bool is_sacred_context {};
    int32_t gpu_sync {500};
  protected:
    /* detection */
    bicudo::gpu_rm_divine_pipeline_t pipeline_collision_detection {
      .tag = "collision-detection", .description = "Where the collision detection is performed."
    };
  public:
    rocm(bicudo::rocm_pipeline_configuration_t pipeline_config) : base() {
      this->pipeline_config = pipeline_config;
    }
  public:
    bicudo::result_t init() override;
    bicudo::result_t registry_hypergroup(bicudo::hypergroup_t *p_hypergroup) override;
    bicudo::result_t registry_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) override;
    bicudo::result_t unregistry_hypergroup(bicudo::hypergroup_t *p_hypergroup) override;
    bicudo::result_t unregistry_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) override;
    bicudo::result_t update_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) override;
    bicudo::result_t update(bicudo::physics_update_mode mode) override;
    bicudo::result_t size_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body, bicudo::vec2_t<float> size) override;
    bicudo::result_t move_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body, bicudo::vec2_t<float> direction) override;
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
  protected:
    void update_hypergroup_sacred_atomic_machine(
      bicudo::hypergroup_t &hypergroup
    );
  };
}

#endif
