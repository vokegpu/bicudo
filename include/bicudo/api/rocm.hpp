#ifndef BICUDO_API_ROCM_HPP
#define BICUDO_API_ROCM_HPP

#include "bicudo/gpu/rocm_platform.hpp"
#include "bicudo/gpu/rocm_model.hpp"
#include "bicudo/bicudo.hpp"
#include "bicudo/gpu/algebra_linear.hpp"
#include "bicudo/physics/placement.hpp"
#include "base.hpp"

namespace bicudo::api {
  struct rocm : public bicudo::api::base {
  protected:
    bicudo::gpu::rocm_pipeline pipeline {};
    bicudo::gpu::packed_collision_info_and_two_rect detect_collision_memory {};
    uint64_t detect_collision_memory_index {};
  protected:
    void compute_detect_collision_kernel(
      bicudo::physics::collision_info_t *p_collision_info,
      bicudo::physics::placement *&p_a,
      bicudo::physics::placement *&p_b
    );

    void collision_info_memory_fetch(
      bicudo::gpu::packed_collision_info_and_two_rect *p_packed,
      bicudo::physics::collision_info_t *p_collision_info,
      bicudo::types op_type
    );

    void rect_a_b_memory_writestore(
      bicudo::gpu::packed_collision_info_and_two_rect *p_packed,
      bicudo::physics::placement *&p_a,
      bicudo::physics::placement *&p_b
    );
  public:
    void init() override;
    void quit() override;
    void update_physics_simulator(
      bicudo::physics::placement *&p_a,
      bicudo::physics::placement *&p_b,
      bicudo::physics::collision_info_t *p_collision_info
    ) override;
  };
}

#endif