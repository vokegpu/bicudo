#ifndef MEOW_GPU_RENDERER_HPP
#define MEOW_GPU_RENDERER_HPP

#include <GL/glew.h>
#include <vector>
#include <unordered_map>
#include <string_view>

#include "bicudo/gpu/types.hpp"

namespace meow::gpu {
  struct shader {
  public:
    uint32_t stage {};
    const char *p_path_or_source {};
  };

  struct draw_call_t {
  public:
    uint32_t polygon_type {};
    uint32_t index_type {};
    bicudo::types mode {};
    uint32_t vao {};
    uint64_t size {};
    uint64_t offset {};
    std::vector<uint32_t> buffers {};
  };

  struct uniform {
  public:
    uint32_t linked_program {};
    std::unordered_map<std::string_view, uint32_t> program_location_map {};
  public:
    void registry(std::string_view key);
    uint32_t &operator[](std::string_view uniform);
  };
}

namespace meow {
  bicudo::result gpu_compile_shader_program(
    uint32_t *p_program,
    const std::vector<meow::gpu::shader> &shader_list
  );

  bicudo::result gpu_dispatch_draw_call(
    meow::gpu::draw_call_t *p_draw_call
  );
}

#endif