#ifndef BICUDO_GPU_MODEL_HPP
#define BICUDO_GPU_MODEL_HPP

#include "types.hpp"

#include <hip/hip_runtime.h>
#include <hip/hiprtc.h>

namespace bicudo::gpu {
  struct buffer {
  public:
    uint64_t size {};
    void *p_data {};
  };

  struct function {
  public:
    const char *p_entry_point {};
    dim3 grid {};
    dim3 block {};
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
    std::vector<ekg::gpu::function> function_list {};
  };

  struct pipeline {
  public:
    std::vector<bicudo::gpu::kernel> kernel_list {}; 
  };

  typedef pipeline_create_info pipeline;
}

namespace bicudo {
  bicudo::result dispatch(
    bicudo::gpu::pipeline *p_pipeline,
    uint64_t module_index,
    uint64_t kernel_index
  );

  bicudo::result writeback(
    bicudo::gpu::pipeline *p_pipeline,
    uint64_t module_index,
    uint64_t kernel_index
  );

  bicudo::result create_pipeline(
    bicudo::pipeline *p_pipeline,
    bicudo::pipeline_create_info *p_pipeline_create_info
  );
}

#endif