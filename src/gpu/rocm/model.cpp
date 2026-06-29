#include <bicudo/gpu/rocm/model.hpp>

bicudo::result_t bicudo::rocm::init() {
  bicudo::device_id_t device_count {};
  if (hipGetDeviceCount(&device_count) != hipSuccess) {
    bicudo::loge("Failed to initialize ROCm pipeline; no device found");
    return bicudo::result::FAILED_TO_INITIALIZE_ROCM;
  }

  bicudo::log("HIP runtime found ", device_count, " devices!");

  for (bicudo::device_id_t i {}; i < device_count; i++) {
    hipDeviceProp_t deviceProp {};
    if (hipGetDeviceProperties(&deviceProp, i) != hipSuccess) {
      bicudo::logw("HIP runtime could not get device properties; device ", i, " !");
      continue;
    }

    bicudo::log("Device name: ", deviceProp.name);
  }

  if (hipSetDevice(pipeline_config.set_order) != hipSuccess) {
    bicudo::log("HIP runtime could not set interal device; check device set order to be zero or an existent device id!");
    return bicudo::result::FAILED_TO_INITIALIZE_ROCM;
  }

  bicudo::log("Bicudo selected device ID ", pipeline_config.set_order);

  return bicudo::result::SUCCESS;
}
