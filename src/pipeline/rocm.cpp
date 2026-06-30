#include <bicudo/pipeline/rocm.hpp>

bicudo::pipeline::base *bicudo::as_rocm(
  bicudo::rocm_pipeline_configuration_t pipeline_config
) {
  return new bicudo::rocm(pipeline_config);
}
