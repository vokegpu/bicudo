#include <bicudo/bicudo.hpp>

bicudo::result_t bicudo::init(
  bicudo::init_core_t &init_core,
  bicudo::core_t &core
) {
  if (!init_core.p_base) {
    bicudo::loge("Failed to initialize Bicudo core; no pipeline base was set!");
    return bicudo::result::FAILED_TO_INITIALIZE_BICUDO;
  }

  core.p_base = init_core.p_base;
  if (core.p_base->init() == bicudo::result::FAILED_TO_INITIALIZE_ROCM) {
    bicudo::loge("Failed to initialize Bicudo core; failed to initialize ROCm!");
    return bicudo::result::FAILED_TO_INITIALIZE_BICUDO;
  }

  bicudo::log("Successfully initalized Bicudo core!");
  return bicudo::result::OK;
}
