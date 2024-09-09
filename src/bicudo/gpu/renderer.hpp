#ifndef BICUDO_GPU_RENDERER_HPP
#define BICUDO_GPU_RENDERER_HPP

#include <GL/glew.h>
#include <vector>
#include <unordered_map>
#include <string_view>

#include "types.hpp"

namespace bicudo::gpu {
  struct shader {
  public:
    uint32_t stage {};
    const char *p_path_or_source {};
  };

  struct draw_call {
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
    uint32_t &operator[](std::string_view uniform);
  };
}

namespace bicudo {
  bicudo::result gpu_compile_shader_program(
    uint32_t *p_program,
    const std::vector<bicudo::gpu::shader> &shader_list
  );

  bicudo::result gpu_dispatch_draw_call(
    bicudo::gpu::draw_call *p_draw_call
  );
}

#endif