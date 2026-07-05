#ifndef BICUDO_PHYSICS_HYPERGROUP_HPP
#define BICUDO_PHYSICS_HYPERGROUP_HPP

#include <bicudo/io/signature.hpp>
#include <bicudo/physics/body.hpp>
#include <bicudo/gpu/gpu.hpp>
#include <vector>
#include <chrono>

namespace bicudo {
  struct hypergroup_t {
  public:
    /**
     * This is reserved for GPU-accelerated
     * atomic operations.
     **/

    bicudo::gpu_sacred_atomic_memory_t atomic {};
    bicudo::hypergroup_state state = bicudo::hypergroup_state::UNHANDLED;
    bicudo::gpu_sacred_dispatch_params_t ref_params {};
    bicudo::gpu_sacred_param_pointers_t raw_params {}; 

    bool has_host_memory_synced {};
    bool should_host_memory_be_synced {};

    std::size_t bytes_stride_gpu_pass {};
    std::size_t body_count_per_gpu_wave_block {};
    bicudo::vec3_t<uint32_t> block_runnings {};
    bicudo::vec3_t<uint32_t> grid_runnings {};

    std::chrono::steady_clock::time_point elapsed {};
  public:
    std::string tag {};
    std::string description {};
    std::vector<bicudo::body_t*> bodies {};
  public:
    bicudo_as_signed(bicudo::hypergroup_t);
  };
}

#endif
