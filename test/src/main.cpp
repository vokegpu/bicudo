#include "meow.hpp"
#include "world/pickup.hpp"

meow::application_t meow::app {}; 

#if defined(BICUDO_HIP_ROCM)
  #include <bicudo/pipeline/rocm.hpp>
  #define BASE bicudo::as_rocm()
#else
  #include <bicudo/pipeline/cpu.hpp>
  #define BASE bicudo::as_cpu()
#endif

void init_ekg() {
  ekg::bind("click-on-object", "mouse-1");
  ekg::bind("drop-object", "mouse-1-up");
  ekg::bind("click-on-camera", "mouse-2");
  ekg::bind("drop-camera", "mouse-2-up");
  ekg::bind("zoom-camera", "mouse-wheel");
}

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

  init_ekg();

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

  bicudo::hypergroup_t hypergroup {};
  bicudo::registry(&hypergroup);

  bicudo::body_t a {.pos = {420, 400}, .size = {100, 100}, .angle = 0.0};
  bicudo::registry(&hypergroup, &a);

  bicudo::body_t b {.pos = {570, 417}, .size = {100, 100}, .unique_id = 2};
  bicudo::registry(&hypergroup, &b);

  bicudo::vec4_t<float> body_color {};
  ekg::input_info_t &input = ekg::input();

  while (meow::app.running) {
    while (SDL_PollEvent(&sdl_event)) {  
      ekg::sdl2_poll_event(sdl_event);

      meow::tools_pick_camera(
        meow::app.camera_pickup
      );

      meow::tools_pick_object_from_world(
        hypergroup,
        meow::app.global_body_pickup
      );

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

    meow::tools_update_picked_camera(
      meow::app.camera_pickup
    );

    meow::tools_update_picked_object(
      hypergroup,
      meow::app.global_body_pickup
    );

    bicudo::update(bicudo::physics_update_mode::EVERYTHING);

    glClearColor(clear_color.x, clear_color.y, clear_color.z, clear_color.w);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport(0.0f, 0.0f, ekg::dpi.viewport.w, ekg::dpi.viewport.h);   

    meow::app.immediate.invoke();

    for (bicudo::body_t *p_body : hypergroup.bodies) {
      bicudo::update(
        &hypergroup,
        p_body
      );

      bicudo::vec4_t<float> rect_on_camera {
        p_body->rect
      };

      rect_on_camera.x -= meow::app.camera.rect.pos.x;
      rect_on_camera.y -= meow::app.camera.rect.pos.y;

      meow::app.immediate.draw(
        rect_on_camera,
        p_body->has_collide ? bicudo::vec4_t<float>(0.8f, 0.7, 0.8f, 1.0f) : bicudo::vec4_t<float>(0.8f, 0.7, 0.8f, 0.5f),
        p_body->angle, 0
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
