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
  ekg::bind("move-object", "mouse-1");
  ekg::bind("drop-object", "mouse-1-up");
  ekg::bind("options-object", "mouse-3");
  ekg::bind("click-on-camera", "mouse-1");
  ekg::bind("world-popup", "mouse-3");
  ekg::bind("drop-camera", "mouse-1-up");
  ekg::bind("zoom-camera", "mouse-wheel");

  ekg::make<ekg::stack_t>(
    {
      .tag = "in-world-UIs"
    }
  );

  ekg::make<ekg::frame_t>({.rect = {.w = 252, .h = 252}, .resize = ekg::dock::left | ekg::dock::bottom | ekg::dock::right});
  ekg::make<ekg::label_t>({.text = "-- Bicudo Physics Engine v0.1", .dock = ekg::dock::fill | ekg::dock::next});
  ekg::make<ekg::label_t>({.text = &meow::app.gui.stats_body_count, .dock = ekg::dock::fill | ekg::dock::next});
  ekg::make<ekg::label_t>({.text = "-- " + bicudo::device(), .dock = ekg::dock::fill | ekg::dock::next});
  ekg::make<ekg::label_t>({.text = &meow::app.gui.stats_framerate, .dock = ekg::dock::fill | ekg::dock::next});
  ekg::make<ekg::label_t>({.text = &meow::app.gui.stats_grid, .dock = ekg::dock::fill | ekg::dock::next});
  ekg::make<ekg::label_t>({.text = &meow::app.gui.stats_block, .dock = ekg::dock::fill | ekg::dock::next});
  ekg::make<ekg::label_t>({.text = "-- 52~", .dock = ekg::dock::fill | ekg::dock::bottom});
  ekg::pop<ekg::frame_t>();

  ekg::popup_t &in_world_popup = ekg::make<ekg::popup_t>({.tag = "in-world-popup"});
  meow::app.gui.in_world_popup = in_world_popup;

  ekg::button_t button {
    .dock = ekg::dock::next | ekg::dock::fill
  };

  button.tag = "reset-camera";
  ekg::button_t::check_t &content = button.checks.emplace_back();

  content.text = "Reset Camera";
  content.actions[ekg::action::press] = ekg::make<ekg::callback_t>(
    {
      .info = {.tag = "reset-camera"},
      .lambda = [](ekg::info_t&) {
        meow::app.camera.rect.pos.x = 0.0f;
        meow::app.camera.interpolated_zoom = 1.0f;
        meow::app.camera.rect.pos.y = 0.0f;
        meow::app.camera.zoom = 1.0f;
        meow::app.camera.rect.velocity = {};
        meow::app.immediate.current_zoom = 1.0f;

        meow::app.immediate.set_viewport(
          meow::app.immediate.viewport.z,
          meow::app.immediate.viewport.w
        );
      }
    }
  );

  ekg::make<ekg::button_t>(button);

  ekg::pop<ekg::popup_t>();
  ekg::pop<ekg::stack_t>();
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

  bicudo::init_core_t bicudo_init_core = {
    .p_base = BASE
  };

  bicudo::init(
    bicudo_init_core,
    meow::app.bicudo
  );

  ekg::rgba_t<float> clear_color(0.0f, 0.0f, 0.0f, 1.0f);

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

  ekg::timing_t framerate {};
  int32_t last_frame_count {1};
  int32_t elapsed_frame_count {};

  meow::app.immediate.create();

  bicudo::hypergroup_t hypergroup {.tag = "in-editor-physics-hypergroup"};
  bicudo::registry(&hypergroup);

  std::size_t bodies_in_scene {256};
  std::vector<bicudo::body_t> bodies {};
  bodies.resize(bodies_in_scene);

  //std::srand(std::time({}));

  for (std::size_t j {}; j < bodies_in_scene; j++) {
    auto x = std::rand() % 2000;
    auto y = std::rand() % 2000;

    auto w = std::rand() % 300;
    auto h = std::rand() % 300;

    bicudo::body_t &body = bodies.emplace_back();
    body.pos.x = x;
    body.pos.y = y;
    body.size.x = w;
    body.size.y = h;
    bicudo::registry(&hypergroup, &body);
  }

  //bicudo::body_t b {.pos = {570, 417}, .size = {100, 100}, .mass = 2.0f};
  //bicudo::registry(&hypergroup, &b);

  bicudo::vec4_t<float> body_color {};
  ekg::input_info_t &input = ekg::input();

  float vel {};

  meow::app.immediate.uinf = 0.000520f + 0.01977f + 0.0372f + 0.01999f + 0.0153f;

  while (meow::app.running) {
    while (SDL_PollEvent(&sdl_event)) {  
      ekg::sdl2_poll_event(sdl_event);

      meow::tools_pick_object_from_world(
        hypergroup,
        meow::app.global_body_pickup
      );

      meow::tools_pick_camera(
        meow::app.camera_pickup
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

      meow::app.gui.stats_framerate = "fps: " + std::to_string(last_frame_count);
      meow::app.gui.stats_body_count = "body(s): " + std::to_string(hypergroup.bodies.size());
      
      meow::app.gui.stats_grid = (
        "grid: "
        +
        (
          std::to_string(hypergroup.grid_runnings.x)
          + " , " +
          std::to_string(hypergroup.grid_runnings.y)
          + ", " +
          std::to_string(hypergroup.grid_runnings.z)
        )
      );

      meow::app.gui.stats_block = (
        "block: "
        +
        (
          std::to_string(hypergroup.block_runnings.x)
          + " , " +
          std::to_string(hypergroup.block_runnings.y)
          + ", " +
          std::to_string(hypergroup.block_runnings.z)
        )
      );
      
      ekg::gui.ui.redraw = true;

      //ekg::log::flush();
      bicudo::flush();
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

    meow::app.immediate.draw(
      bicudo::vec4_t<float>(0.0f, 0.0f, meow::app.camera.view.z, meow::app.camera.view.w),
      bicudo::vec4_t<float>(0.052f, 0.052f, 0.052f, 0.052f),
      0, 1
    );
    
    auto hex = bicudo::vec4_t<float>(0.0f, 0.0f, meow::app.camera.view.z, meow::app.camera.view.w);
    hex.z /= (7 * meow::app.camera.zoom);
    hex.w = hex.z;

    //hex.z += (sin(vel) * (meow::app.camera.view.w / 3));
    //hex.w += (cos(vel) * (meow::app.camera.view.w / 3));
 
    hex.x = meow::app.camera.view.z / 2 - (hex.z / 2);
    hex.y = meow::app.camera.view.w / 2 - (hex.w / 2);

    meow::app.immediate.draw(
      hex,
      bicudo::vec4_t<float>(0.052f, 0.052f, 0.052f, 0.052f),
      45.0f - meow::app.immediate.uinf - vel, 2
    );

    hex.z /= 3;
    hex.w = hex.z;

    hex.x = meow::app.camera.view.z / 2 - (hex.z / 2);
    hex.y = meow::app.camera.view.w / 2 - (hex.w / 2);

    meow::app.immediate.draw(
      hex,
      bicudo::vec4_t<float>(0.052f, 0.052f, 0.052f, 0.052f),
      0.0f + meow::app.immediate.uinf + vel, 3
    );

    vel += 2.0f;

    meow::app.immediate.viewport.z = ekg::dpi.viewport.w;
    meow::app.immediate.viewport.w = ekg::dpi.viewport.h;

    for (bicudo::body_t *p_body : hypergroup.bodies) {
      //bicudo::update(
      //  &hypergroup,
      //  p_body
      //);

      bicudo::vec4_t<float> frustum {
        meow::app.camera.rect.pos.x,
        meow::app.camera.rect.pos.y,
        meow::app.immediate.viewport.z,
        meow::app.immediate.viewport.w
      };

      if (!bicudo::aabb_collide_with_aabb(meow::app.camera.view, p_body->rect)) {
        //continue;
      }

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

  return bicudo::flush();
}
