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

    std::size_t body_byte_index {};
    bool has_memory_filled_once {};
    std::size_t bodies_pass_count {};
    std::chrono::steady_clock::time_point elapsed {};
    bool refresh {};
  public:
    std::string tag {};
    std::string description {};
    std::vector<bicudo::body_t*> bodies {};
  public:
    bicudo_as_signed(bicudo::hypergroup_t);
  };
}

#endif
