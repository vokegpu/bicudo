#include <bicudo/pipeline/rocm.hpp>

bicudo::pipeline::base *bicudo::rocm(bicudo::rocm_pipeline_configuration_t pipeline_config) {
  return new bicudo::pipeline::rocm(pipeline_config);
}
