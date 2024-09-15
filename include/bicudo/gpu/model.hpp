#ifndef BICUDO_GPU_MODEL_HPP
#define BICUDO_GPU_MODEL_HPP

#include "types.hpp"
#include "rocm.hpp"

#include <cstdint>
#include <vector>

namespace bicudo::gpu {
  struct buffer {
  public:
    uint64_t size {};
    void *p_device {};
    void *p_host {};
  };

  struct function {
  public:
    const char *p_entry_point {};
    dim3 grid {};
    dim3 block {};
    size_t shared_mem_bytes {};
    hipStream_t stream {};
    std::vector<bicudo::gpu::buffer> buffer_list {};
    hipFunction_t entry_point {};
    std::vector<void*> argument_list {};
    uint64_t mem_size {};
  };

  struct kernel {
  public:
    const char *p_tag {};
    const char *p_src {};    
    hipModule_t module {};
    hiprtcProgram program {};
    std::vector<bicudo::gpu::function> function_list {};
  };

  struct pipeline {
  public:
    const char *p_tag {};
    std::vector<bicudo::gpu::kernel> kernel_list {}; 
  };

  typedef bicudo::gpu::pipeline pipeline_create_info;
}

namespace bicudo {
  bicudo::result gpu_dispatch(
    bicudo::gpu::pipeline *p_pipeline,
    uint64_t module_index,
    uint64_t kernel_index
  );

  bicudo::result gpu_writeback(
    bicudo::gpu::pipeline *p_pipeline,
    uint64_t module_index,
    uint64_t kernel_index,
    uint64_t param_index
  );

  bicudo::result gpu_create_pipeline(
    bicudo::gpu::pipeline *p_pipeline,
    bicudo::gpu::pipeline_create_info *p_pipeline_create_info
  );
}

#endif