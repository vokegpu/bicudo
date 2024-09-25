#ifndef BICUDO_GPU_ROCM_MODEL_HPP
#define BICUDO_GPU_ROCM_MODEL_HPP

#include "types.hpp"
#include "rocm_platform.hpp"
#include "content.hpp"

#include <cstdint>
#include <vector>

namespace bicudo::gpu {
  struct rocm_function {
  public:
    const char *p_entry_point {};
    dim3 grid {};
    dim3 block {};
    size_t shared_mem_bytes {};
    hipStream_t stream {};
    std::vector<bicudo::gpu::buffer_t> buffer_list {};
    hipFunction_t entry_point {};
    std::vector<void*> argument_list {};
    uint64_t mem_size {};
  };

  struct rocm_kernel {
  public:
    const char *p_tag {};
    const char *p_src {};    
    hipModule_t module {};
    hiprtcProgram program {};
    std::vector<bicudo::gpu::rocm_function> function_list {};
  };

  struct rocm_pipeline {
  public:
    const char *p_tag {};
    std::vector<bicudo::gpu::rocm_kernel> kernel_list {}; 
  };

  typedef bicudo::gpu::rocm_pipeline rocm_pipeline_create_info;
}

namespace bicudo {
  bicudo::result gpu_rocm_dispatch(
    bicudo::gpu::rocm_pipeline *p_pipeline,
    uint64_t module_index,
    uint64_t kernel_index
  );

  bicudo::result gpu_rocm_memory_fetch(
    bicudo::gpu::rocm_pipeline *p_pipeline,
    uint64_t module_index,
    uint64_t kernel_index,
    uint64_t param_index,
    bicudo::types op_type
  );

  bicudo::result gpu_rocm_create_pipeline(
    bicudo::gpu::rocm_pipeline *p_pipeline,
    bicudo::gpu::rocm_pipeline_create_info *p_pipeline_create_info
  );
}

#endif