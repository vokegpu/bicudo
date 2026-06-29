#ifndef BICUDO_GPU_ROCM_DIVINE_HPP
#define BICUDO_GPU_ROCM_DIVINE_HPP

#include <bicudo/gpu/rocm/header.hpp>
#include <bicudo/math/geometry.hpp>
#include <bicudo/log/log.hpp>

#include <string>
#include <vector>

namespace bicudo {
  struct gpu_rm_divine_fun_memory_properties_t {
  public:
    std::size_t shared_mem_bytes {};
    std::size_t mem_size {};
  public:
    hipStream_t stream {};
  };
  
  struct gpu_rm_divine_fun_dimension_t {
  public:
    bicudo::vec3_t<uint32_t> grid {};
    bicudo::vec3_t<uint32_t> block {};
  };
  
  struct gpu_rm_divine_fun_entry_point_t {
  public:
    std::string name {};
  public:
    hipFunction_t h_fun {};
  };

  struct gpu_rm_divine_fun_args_t {
  public:
    std::string tag {};
    std::size_t bytes {};
    void *p_host {};
    void *p_device {};
  };

  using gpu_rm_divine_fun_arguments_t = std::vector<bicudo::gpu_rm_divine_fun_args_t>;

  struct gpu_rm_divine_fun_t {
  public:
    bicudo::gpu_rm_divine_fun_entry_point_t entry_point {};
    bicudo::gpu_rm_divine_fun_memory_properties_t memory {};
    bicudo::gpu_rm_divine_fun_dimension_t dimension {};
    bicudo::gpu_rm_divine_fun_arguments_t args {};
  };

  using gpu_rm_divine_functions_t = std::vector<bicudo::gpu_rm_divine_fun_t>;

  struct gpu_rm_divine_kernel_t {
  public:
    std::string tag {};
    std::string src {};
    bicudo::gpu_rm_divine_functions_t functions {};
    bicudo::result_t status {bicudo::result::KERNEL_NOT_INITIALIZED};
  public:
    hipModule_t hip_module {};
    hiprtcProgram hip_program {};
  };

  using gpu_rm_divine_kernels_t = std::vector<bicudo::gpu_rm_divine_kernel_t>;
  using gpu_rm_divine_module_t = bicudo::gpu_rm_divine_kernel_t;

  struct gpu_rm_divine_pipeline_t {
  public:
    std::string tag {};
    std::string description {};
    bicudo::gpu_rm_divine_kernels_t kernels {};
  };
}

namespace bicudo {
  bicudo::gpu_rm_divine_kernel_t &as_kernel(bicudo::gpu_rm_divine_pipeline_t &pipeline);
}

#endif
