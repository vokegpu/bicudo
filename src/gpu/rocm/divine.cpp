#include <bicudo/gpu/rocm/divine.hpp>

bicudo::gpu_rm_divine_kernel_t &bicudo::as_kernel(bicudo::gpu_rm_divine_pipeline_t &pipeline) {
 return pipeline.kernels.emplace_back();
}
