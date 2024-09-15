#ifndef BICUDO_GRAPHICS_IMMEDIATE_HPP
#define BICUDO_GRAPHICS_IMMEDIATE_HPP

#include "gpu/renderer.hpp"
#include "bicudo/util/math.hpp"

namespace meow {
  class immediate_graphics {
  protected:
    uint32_t program {};
    meow::gpu::uniform uniform {};
    meow::gpu::draw_call_t draw_call {};
    bicudo::mat4 mat4x4_projection {};
    bicudo::mat4 mat4x4_rotate {};
    bicudo::vec4 viewport {};
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
      float angle,
      uint32_t bind_texture = 0
    );
  };
}

#endif