#include <cstdint>

#include <bicudo/bicudo.hpp>
#include <bicudo/pipeline/rocm.hpp>
#include <bicudo/gpu/rocm/programs.hpp>

int32_t main(int32_t, char**) {
  bicudo::init_core_t bicudo_init_core = {
    .p_base = bicudo::as_rocm()
  };

  bicudo::core_t core {};
  bicudo::init(bicudo_init_core, core);

  bicudo::gpu_rm_sacred_atomic_memory_t<float, float> atomic {
    .host = {
      3.0f, // should be 17.0f
      4.0f, // should be 27.0f
      5.0f, // should be 37.0f
      6.0f, // should be 47.0f
      7.0f  // should be 52.0f
    },
    .device = std::vector<float>(5),
    .bytes = sizeof(float)*5
  };

  bicudo::gpu_rm_divine_kernel_t kernel_hip_runtime {
    .tag = "hip-runtime-assert",
    .src = bicudo::gpu::rocm::hip_rocm_kernel_runtime_assert
  };

  kernel_hip_runtime.functions = {
    {
      .entry_point = {
        .name = "runtime_assert_entrypoint"
      },
      .memory = {
        .shared_mem_bytes = 0,
        .mem_size = 0,
        .stream = nullptr,
      },
      .dimension = {
        .grid = bicudo::vec3_t<uint32_t>(1, 1, 1),
        .block = bicudo::vec3_t<uint32_t>(4, 1, 1)
      },
      .args = {
        {
          .bytes = atomic.bytes,
          .p_host = atomic.host.data(),
          .p_device = atomic.device.data()
        }
      }
    }
  };

  bicudo::gpu_rm_divine_pipeline_t pipeline { .tag = "52" };
  pipeline.kernels.push_back(kernel_hip_runtime);

  bicudo::rocm &rocm {bicudo::as_gpu<bicudo::rocm>()};  

  return bicudo::flush();
}
