#ifndef BICUDO_GPU_ROCM_SACRED_HPP
#define BICUDO_GPU_ROCM_SACRED_HPP

#include <cstdint>
#include <vector>

namespace bicudo {
  template<typename t, typename s>
  struct gpu_rm_sacred_atomic_memory_t {
  public:
    std::vector<t> host {};
    std::vector<s> device {};
    std::size_t bytes {};
  };
}

#endif
