#include <iostream>
#include "bicudo/bicudo.hpp"

#include <ekg/ekg.hpp>
#include <ekg/os/ekg_opengl.hpp>
#include <ekg/os/ekg_sdl.hpp>

#include <GL/glew.h>
#include <SDL2/SDL.h>

__global__ void meow(uint32_t *p_oi) {
  // ...
}

__global__ void mu(uint32_t *p_oi) {
  // ...
}

int32_t main(int32_t, char**) {
  uint32_t number {};

  bicudo::gpu::pipeline pipeline {};
  bicudo::gpu_create_pipeline(
    &pipeline,
    &pipeline_create_info
  );

  SDL_Init(SDL_INIT_VIDEO);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4);

  SDL_Window *p_sdl_win {
    SDL_CreateWindow(
      "oiiii",
      10,
      10,
      800,
      600,
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    )
  };

  SDL_GLContext sdl_gl_context {
    SDL_GL_CreateContext(p_sdl_win)
  };

  glewExperimental = GL_TRUE;
  glewInit();

  ekg::runtime_property ekg_runtime_property {
    .p_font_path = "./whitneybook.otf",
    .p_font_path_emoji = "twemoji.ttf",
    .p_gpu_api = new ekg::os::opengl(),
    .p_os_platform = new ekg::os::sdl(p_sdl_win)
  };

  ekg::runtime runtime {};
  ekg::init(
    &runtime,
    &ekg_runtime_property
  );

  ekg::frame("oiii muuu", {20, 20}, {200, 200})
    ->set_resize(ekg::dock::left | ekg::dock::bottom | ekg::dock::right | ekg::dock::top)
    ->set_drag(ekg::dock::full);

  uint32_t number {};

  ekg::button("couwnt in GPU:")
    ->set_task(
      new ekg::task {
        .info = {
          .tag = "omg gpu kkkk"
        },
        .function = [&number](ekg::info &info) {
          bicudo::count(&number);
        }
      },
      ekg::action::activity
    );

  ekg::slider<uint32_t>("meow-gpu", ekg::dock::fill)
    ->set_text_align(ekg::dock::left | ekg::dock::center)
    ->range<uint32_t>(0, 0, 0, 100)
    ->range<uint32_t>(0).u32.transfer_ownership(&number);

  ekg::pop_group();

  SDL_Event sdl_event {};

  bool running {true};
  while (running) {
    while (SDL_PollEvent(&sdl_event)) {
      if (sdl_event.type == SDL_QUIT) {
        running = false;
      }

      ekg::os::sdl_poll_event(sdl_event);
    }

    ekg::ui::dt = 0.016f;
    ekg::update();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(0.7f, 0.3f, 0.8f, 1.0f);
    glViewport(0.0f, 0.0f, ekg::ui::width, ekg::ui::height);

    ekg::render();

    SDL_GL_SwapWindow(p_sdl_win);
    SDL_Delay(16);
  }

  return 0;
}