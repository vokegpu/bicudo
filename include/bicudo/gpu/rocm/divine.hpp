#ifndef BICUDO_GPU_ROCM_DIVINE_HPP
#define BICUDO_GPU_ROCM_DIVINE_HPP

#include <bicudo/gpu/rocm/header.hpp>
#include <bicudo/math/geometry.hpp>
#include <bicudo/io/signature.hpp>

#include <string>
#include <vector>

namespace bicudo {
  using gpu_rm_sacred_pointers_t = std::vector<void*>;

  struct gpu_rm_divine_fun_memory_properties_t {
  public:
    std::size_t shared_mem_bytes {};
    bicudo::gpu_rm_sacred_pointers_t sacred_pointers {};
    std::size_t sacred_pointers_mem_bytes_length {};
  public:
    hipStream_t h_stream {};
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
    void *p_device {};
  };

  using gpu_rm_divine_fun_arguments_t = std::vector<bicudo::gpu_rm_divine_fun_args_t>;

  struct gpu_rm_divine_fun_t {
  public:
    bicudo::gpu_rm_divine_fun_entry_point_t entry_point {};
    bicudo::gpu_rm_divine_fun_memory_properties_t memory {};
    bicudo::gpu_rm_divine_fun_dimension_t dimension {};
    bicudo::gpu_rm_divine_fun_arguments_t args {};
  public:
    bicudo_as_signed(bicudo::gpu_rm_divine_fun_t);
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
  public:
    bicudo_as_signed(bicudo::gpu_rm_divine_kernel_t);
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
  bicudo::gpu_rm_divine_pipeline_t &as_new_pipeline();

  bicudo::gpu_rm_divine_kernel_t &as_kernel(
    bicudo::gpu_rm_divine_pipeline_t &pipeline
  );

  bicudo::gpu_rm_divine_module_t &as_module(
    bicudo::gpu_rm_divine_pipeline_t &pipeline,
    std::size_t index
  );

  bicudo::gpu_rm_divine_module_t &as_module(
    bicudo::gpu_rm_divine_pipeline_t &pipeline,
    const std::string &tag
  );

  bicudo::gpu_rm_divine_fun_t &as_function(
    bicudo::gpu_rm_divine_module_t &kmodule,
    std::size_t index
  );

  bicudo::gpu_rm_divine_fun_t &as_function(
    bicudo::gpu_rm_divine_module_t &kmodule,
    const std::string &name
  );
}

#endif
