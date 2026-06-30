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

  bicudo::rocm &rocm = bicudo::as_gpu<bicudo::rocm>();
  bicudo::gpu_rm_divine_pipeline_t &pipeline52 = rocm.gpu_pipeline_new();

  pipeline52 = {
    .tag = "52", .description = "The divine kernel for Divine numbers assertation."
  };

  bicudo::gpu_rm_sacred_atomic_memory_t atomic(sizeof(float)*5);
  bicudo::gpu_allocate_sacred_atomic(atomic, hipHostMallocMapped, 0);;

  bicudo::gpu_rm_divine_kernel_t &kernel_hip_runtime = bicudo::as_kernel(pipeline52);

  kernel_hip_runtime = {
    .tag = "hip-runtime-assert",
    .src = bicudo::gpu::rocm::hip_rocm_kernel_runtime_assert
  };

  kernel_hip_runtime.functions = {
    {
      .entry_point = {
        .name = "runtime_assert_entry_point"
      },
      .memory = {
        .h_stream = nullptr
      },
      .dimension = {
        .grid = bicudo::vec3_t<uint32_t>(1, 1, 1),
        .block = bicudo::vec3_t<uint32_t>(1, 1, 1)
      },
      .args = {
        {
          .tag = "assert-buf",
          .bytes = atomic.bytes,
          .p_device = atomic.p_device
        }
      }
    }
  };

  rocm.gpu_pipeline_load_kernels(pipeline52);
  rocm.gpu_pipeline_create(pipeline52);

  bicudo::gpu_rm_divine_module_t &module_hip_runtime_assert {
    bicudo::as_module(pipeline52, "hip-runtime-assert")
  };

  bicudo::gpu_rm_divine_fun_t &fun_entry_point_hip_runtime_assert {
    bicudo::as_function(module_hip_runtime_assert, "runtime_assert_entry_point")
  };

  if (module_hip_runtime_assert != bicudo::found || fun_entry_point_hip_runtime_assert != bicudo::found) {
    return bicudo::flush();
  }

  float *p = static_cast<float*>(atomic.p_host);

  p[0] = 153.0f;  // should be 17.0f
  p[1] = 26.0f;   // should be 27.0f
  p[2] = 24.0f;   // should be 37.0f
  p[3] = 6.0f;    // should be 47.0f
  p[4] = 1977.0f; // should be 52.0f

  bicudo::gpu_sacred_call(
    module_hip_runtime_assert,
    fun_entry_point_hip_runtime_assert
  );

  bicudo::gpu_sacred_async_fetch(
    fun_entry_point_hip_runtime_assert,
    atomic.p_host,
    atomic.p_device,
    atomic.bytes
  );

  float *j = static_cast<float*>(atomic.p_host);

  for (std::size_t i = 0; i < 5; i++) {
    bicudo::log("-> ", p[i]);
  }

  return bicudo::flush();
}
