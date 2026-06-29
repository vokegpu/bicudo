#include <cstdint>

#include <bicudo/bicudo.hpp>
#include <bicudo/pipeline/rocm.hpp>
#include <bicudo/gpu/rocm/programs.hpp>
#include <bicudo/gpu/gpu.hpp>

int32_t main(int32_t, char**) {
  bicudo::init_core_t bicudo_init_core = {
    .p_base = bicudo::as_rocm()
  };

  bicudo::core_t core {};
  bicudo::init(bicudo_init_core, core);

  bicudo::gpu_rm_sacred_atomic_memory_t<float, float> atomic {
    .host = {
      153.0f,  // should be 17.0f
      26.0f,   // should be 27.0f
      24.0f,   // should be 37.0f
      6.0f,    // should be 47.0f
      1977.0f  // should be 52.0f
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
        .name = "runtime_assert_entry_point"
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

  bicudo::rocm &rocm = bicudo::as_gpu<bicudo::rocm>();
  bicudo::gpu_rm_divine_pipeline_t pipeline52 { .tag = "52", .description = "The divine kernel for Divine numbers assertation." };
  bicudo::gpu_rm_divine_kernels_t kernels = { kernel_hip_runtime };
  
  rocm.gpu_load_kernels(pipeline52, kernels);
  rocm.gpu_create_pipeline(pipeline52);

  return bicudo::flush();
}
