#ifndef MEOW_HPP
#define MEOW_HPP

#include "world/camera.hpp"
#include "world/pickup.hpp"
#include "graphics/graphics.hpp"

#include <bicudo/bicudo.hpp>

#include <ekg/ekg.hpp>
#include <ekg/platform/sdl/sdl2.hpp>
#include <ekg/gpu/opengl/gl.hpp>

namespace meow {
  extern struct application_t {
  public:
    bicudo::core_t bicudo {};
    ekg::runtime_t ekg {};
  public:
    meow::camera camera {};
    meow::immediate_graphics immediate {};
    meow::pickup_info_t global_body_pickup {};
    meow::pickup_info_t camera_pickup {};
  public:
    ekg::ft_library ft_library {};
    SDL_Window *p_sdl_win {};
    bool running {true};
  public:
    bool vsync {true};
  } app;
}

#endif
