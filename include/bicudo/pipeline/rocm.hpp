#ifndef BICUDO_PIPELINE_ROCM_HPP
#define BICUDO_PIPELINE_ROCM_HPP

#include <bicudo/pipeline/rocm/model.hpp>

namespace bicudo {
  bicudo::pipeline::base *rocm(bicudo::rocm_pipeline_configuration_t pipeline_config = {});
}

#endif
