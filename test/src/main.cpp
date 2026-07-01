#include "meow.hpp"

meow::application_t meow::app {}; 

#if defined(BICUDO_HIP_ROCM)
  #include <bicudo/pipeline/rocm.hpp>
  #define BASE bicudo::as_rocm()
#else
  #include <bicudo/pipeline/cpu.hpp>
  #define BASE bicudo::as_cpu()
#endif

int32_t main(int32_t, char**) {
  SDL_Init(SDL_INIT_VIDEO);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
  SDL_GL_SetSwapInterval(0);

  meow::app.p_sdl_win = {
    SDL_CreateWindow(
      "52",
      SDL_WINDOWPOS_CENTERED,
      SDL_WINDOWPOS_CENTERED,
      1280,
      720,
      SDL_WINDOW_RESIZABLE | SDL_WINDOW_OPENGL
    )
  };

  SDL_Event sdl_event {};
  SDL_GLContext sdl_gl_context {SDL_GL_CreateContext(meow::app.p_sdl_win)};
  glewInit();

  ekg::rgba_t<float> clear_color(0.2f, 0.4f, 0.4f, 1.0f);

  FT_Init_FreeType(&meow::app.ft_library);

  ekg::runtime_properties_info_t runtime_properties_info {
    .default_font_path_text = "./comic-mono.ttf",
    .default_font_path_emoji = "./twemoji.ttf",
    .p_platform_base = new ekg::sdl2(meow::app.p_sdl_win),
    .p_gpu_api = new ekg::opengl(),
    .ft_library = meow::app.ft_library
  };

  ekg::init(
    runtime_properties_info,
    meow::app.ekg
  );

  bicudo::init_core_t bicudo_init_core = {
    .p_base = BASE
  };

  bicudo::init(
    bicudo_init_core,
    meow::app.bicudo
  );

  ekg::timing_t framerate {};
  int32_t last_frame_count {1};
  int32_t elapsed_frame_count {};

  meow::app.immediate.create();

  std::vector<bicudo::body_t> bodies {
    {.pos = {420, 400}, .size = {100, 100}, .angle = 0.0, .unique_id = 1},
    {.pos = {570, 417}, .size = {100, 100}, .unique_id = 2}
  };

  bicudo::vec4_t<float> body_color {};
  ekg::input_info_t &input = ekg::input();

  while (meow::app.running) {
    while (SDL_PollEvent(&sdl_event)) {  
      ekg::sdl2_poll_event(sdl_event);

      bool is_up {};
      if (ekg::input("mouse-1") || (is_up = ekg::input("mouse-1-up")) ) {
        for (bicudo::body_t &body : bodies) {
          body.flags = 0;

          if (is_up) continue;

          if (bicudo::vec4_collide_with_vec2(body.rect, {input.interact.x, input.interact.y})) {
            body.delta.x = body.pos.x - input.interact.x;
            body.delta.y = body.pos.y - input.interact.y;
            body.flags = 1;
            break;
          }
        }
      }

      if (sdl_event.type == SDL_WINDOWEVENT && sdl_event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        meow::app.immediate.set_viewport(sdl_event.window.data1, sdl_event.window.data2);
      }

      if (sdl_event.type == SDL_QUIT) {
        meow::app.running = false;
      }
    }

    if (ekg::reset_if_reach(framerate, 1000)) {
      SDL_GL_SetSwapInterval(meow::app.vsync);
      last_frame_count = elapsed_frame_count;
      elapsed_frame_count = 0;
      ekg::log() << last_frame_count;
      
      bicudo::flush();
      ekg::log::flush();

      std::cout << std::flush;
    }

    bicudo::dt = 1.0f / last_frame_count;
    ekg::gui.ui.dt = bicudo::dt;
    ekg::update();

    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0.0f, 0.0f, ekg::dpi.viewport.w, ekg::dpi.viewport.h);   

    meow::app.immediate.invoke();

    bool collided = false;
    bool j = false;

    for (bicudo::body_t &b1 : bodies) {
      if (b1.flags == 1) {
        b1.pos.x = input.interact.x + b1.delta.x;
        b1.pos.y = input.interact.y + b1.delta.y;

        if (input.was_wheel) {
          b1.angle += input.interact.w;
        }
      }

      bicudo::update(b1);

      body_color = {1.0f, 0.6f, 0.8f, 1.0f};
      for (bicudo::body_t &b2 : bodies) {
        if (b1 == b2) continue;
        if (!(bicudo::detect(b1, b2) && bicudo::detect(b2, b1))) continue;
        body_color = {1.0f, 0.0f, 0.8f, 1.0f};
        break;
      }

      meow::app.immediate.draw(
        b1.rect,
        body_color,
        b1.angle, 0
      );
    }

    meow::app.immediate.revoke();

    ekg::render();
    SDL_GL_SwapWindow(meow::app.p_sdl_win);

    if (meow::app.vsync) {
      SDL_Delay(6);
    }
    
    ++elapsed_frame_count;
  }

  ekg::log::flush();
  return bicudo::flush();
}
