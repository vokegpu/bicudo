#ifndef BICUDO_GPU_ROCM_DIVINE_HPP
#define BICUDO_GPU_ROCM_DIVINE_HPP

#include <bicudo/gpu/rocm/header.hpp>
#include <bicudo/math/geometry.hpp>
#include <vector>

namespace bicudo {
  struct gpu_rm_divine_fun_memory_properties_t {
  public:
    std::size_t shared_mem_bytes;
    std::size_t mem_size;
  public:
    hipStream_t stream;
  };
  
  struct gpu_rm_divine_fun_dimension_t {
  public:
    bicudo::vec3_t<uint32_t> grid;
    bicudo::vec3_t<uint32_t> block;
  };
  
  struct gpu_rm_divine_fun_entry_point_t {
  public:
    std::string name;
  public:
    hipFunction_t h_fun;
  };

  struct gpu_rm_divine_fun_args_t {
  public:;
    std::size_t bytes;
    void *p_host;
    void *p_device;
  };

  struct gpu_rm_divine_fun_t {
  public:
    gpu_rm_divine_fun_entry_point_t entry_point {};
    gpu_rm_divine_fun_memory_properties_t memory {};
    gpu_rm_divine_fun_dimension_t dimension {};
    std::vector<gpu_rm_divine_fun_args_t> args {};
  };

  struct gpu_rm_divine_kernel_t {
  public:
    std::string tag {};
    std::string src {};
    std::vector<gpu_rm_divine_fun_t> functions {};
  public:
    hipModule_t hip_module {};
    hiprtcProgram hip_program {};
  };

  struct gpu_rm_divine_pipeline_t {
  public:
    std::string tag {};
    std::string description {};
    std::vector<bicudo::gpu_rm_divine_kernel_t> kernels {};
  };
}

#endif
