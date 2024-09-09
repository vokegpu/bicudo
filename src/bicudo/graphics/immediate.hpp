#ifndef BICUDO_GRAPHICS_IMMEDIATE_HPP
#define BICUDO_GRAPHICS_IMMEDIATE_HPP

#include "bicudo/gpu/renderer.hpp"
#include "bicudo/util/math.hpp"

namespace bicudo {
  class immediate_graphics {
  protected:
    uint32_t program {};
    bicudo::gpu::uniform uniform {};
    bicudo::gpu::draw_call draw_call {};
    bicudo::mat4 mat4x4_projection {};
  public:
    void create();
    void set_viewport(int32_t w, int32_t h);
    void invoke();
    void revoke();

    /**
     * May be not actually idk idc
     **/
    void draw(
      bicudo::vec4 rect,
      bicudo::vec4 color,
      uint32_t bind_texture = 0
    );
  };
}

#endif