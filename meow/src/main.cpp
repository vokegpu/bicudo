#include <iostream>

#include "bicudo/gpu/model.hpp"
#include "bicudo/bicudo.hpp"

#include <ekg/ekg.hpp>
#include <ekg/os/ekg_opengl.hpp>
#include <ekg/os/ekg_sdl.hpp>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include "meow.hpp"

meow::application meow::app {};

void meow::init() {
  bicudo::log() << "Initializing Meow renderer and GUI bindings!";

  meow::app.immediate.create();

  ekg::input::bind("click-on-object", "mouse-1");
  ekg::input::bind("drop-object", "mouse-1-up");
  ekg::input::bind("click-on-camera", "mouse-2");
  ekg::input::bind("drop-camera", "mouse-2-up");
  ekg::input::bind("zoom-camera", "mouse-wheel");
}

void meow::render() {
  bicudo::camera &camera {bicudo::world::camera()};
  meow::app.immediate.invoke();

  bicudo::vec4 rect {};
  bicudo::vec4 color(0.3f, 0.5f, 0.675f, 1.0f);

  meow::app.rendering_placements_count = 0;

  for (bicudo::object *&p_objs : bicudo::app.world_manager.loaded_object_list) {
    rect.x = p_objs->placement.pos.x;
    rect.y = p_objs->placement.pos.y;
    rect.z = p_objs->placement.size.x;
    rect.w = p_objs->placement.size.y;

    if (!bicudo::vec4_collide_with_vec4(rect, camera.rect)) {
      continue;
    }

    rect.x = p_objs->placement.pos.x - camera.placement.pos.x;
    rect.y = p_objs->placement.pos.y - camera.placement.pos.y;

    color.x = p_objs->placement.was_collided;
    meow::app.immediate.draw(rect, color, p_objs->placement.angle);
    meow::app.rendering_placements_count++;

    if (meow::app.settings.show_vertices) {
      meow::app.immediate.draw({p_objs->placement.vertices[0].x - camera.placement.pos.x, p_objs->placement.vertices[0].y - camera.placement.pos.y, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
      meow::app.immediate.draw({p_objs->placement.vertices[1].x - camera.placement.pos.x, p_objs->placement.vertices[1].y - camera.placement.pos.y, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
      meow::app.immediate.draw({p_objs->placement.vertices[2].x - camera.placement.pos.x, p_objs->placement.vertices[2].y - camera.placement.pos.y, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
      meow::app.immediate.draw({p_objs->placement.vertices[3].x - camera.placement.pos.x, p_objs->placement.vertices[3].y - camera.placement.pos.y, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
      meow::app.immediate.draw({rect.x - camera.placement.pos.x + p_objs->placement.size.x / 2, rect.y - camera.placement.pos.y + p_objs->placement.size.y / 2, 10.0f, 10.0f}, {1.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
    }

    if (meow::app.settings.show_aabb) {
      meow::app.immediate.draw({p_objs->placement.min.x - camera.placement.pos.x, p_objs->placement.min.y - camera.placement.pos.y, 10.0f, 10.0f}, {0.0f, 1.0f, 1.0f, 1.0f}, 0.0f);
      meow::app.immediate.draw({p_objs->placement.max.x - camera.placement.pos.x, p_objs->placement.max.y - camera.placement.pos.y, 10.0f, 10.0f}, {0.0f, 0.0f, 1.0f, 1.0f}, 0.0f);
    }
  }

  if (meow::app.settings.show_collision_info) {
    rect.z = 10.0f;
    rect.w = 10.0f;

    rect.x = bicudo::app.world_manager.simulator.collision_info.start.x ;
    rect.y = bicudo::app.world_manager.simulator.collision_info.start.y;

    meow::app.immediate.draw(rect, {1.0f, 0.0f, 0.0f, 1.0f}, 0.0f);

    rect.x = bicudo::app.world_manager.simulator.collision_info.end.x;
    rect.y = bicudo::app.world_manager.simulator.collision_info.end.y;

    meow::app.immediate.draw(rect, {0.0f, 1.0f, 0.0f, 1.0f}, 0.0f);
  }

  meow::app.immediate.revoke();
}

int32_t main(int32_t, char**) {
  SDL_Init(SDL_INIT_VIDEO);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 4);

  SDL_Window *p_sdl_win {
    SDL_CreateWindow(
      "meow 🐈",
      10,
      10,
      800,
      600,
      SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE
    )
  };

  SDL_DisplayMode sdl_display_mode {};
  SDL_GetDisplayMode(0, 0, &sdl_display_mode);

  SDL_SetWindowSize(
    p_sdl_win,
    sdl_display_mode.w - MEOW_INITIAL_WINDOW_OFFSET,
    sdl_display_mode.h - MEOW_INITIAL_WINDOW_OFFSET
  );

  SDL_GLContext sdl_gl_context {
    SDL_GL_CreateContext(p_sdl_win)
  };

  glewExperimental = GL_TRUE;
  glewInit();

  SDL_GL_SetSwapInterval(true);
  bicudo::set_framerate(144);

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

  ekg::theme().set_current_theme_scheme("dark-pinky");

  SDL_Event sdl_event {};
  bool running {true};

  const char *p_shader_count {
    R"(
    extern "C"
    __global__ void meow(uint32_t *p_to_count) {
      if (threadIdx.x == 0) {
        uint32_t &number {*p_to_count};
        number += 2;
      }
    }
    )"
  };

  uint32_t *p_number_device {};
  uint32_t number_host {};

  bicudo::gpu::pipeline_create_info pipeline_create_info {
    .p_tag = "meow",
    .kernel_list =
    {
      {
        .p_tag = "meow?",
        .p_src = p_shader_count,
        .function_list =
        {
          {
            .p_entry_point = "meow",
            .grid = dim3(1, 1, 1),
            .block = dim3(1, 1, 1),
            .shared_mem_bytes = 0,
            .stream = nullptr,
            .buffer_list =
            {
              {
                .size = sizeof(uint32_t),
                .p_device = p_number_device,
                .p_host = &number_host
              }
            }
          }
        }
      }
    }
  };

  bicudo::gpu::pipeline pipeline {};
  bicudo::gpu_create_pipeline(
    &pipeline,
    &pipeline_create_info
  );

  ekg::frame("oiii muuu", {20, static_cast<float>(sdl_display_mode.h) - 700}, {600, 600})
    ->set_resize(ekg::dock::left | ekg::dock::bottom | ekg::dock::right | ekg::dock::top)
    ->set_drag(ekg::dock::full);

  ekg::button("🐄😊💕")
    ->set_task(
      new ekg::task {
        .info = {
          .tag = "omg gpu kkkk"
        },
        .function = [&pipeline](ekg::info &info) {
          bicudo::gpu_dispatch(
            &pipeline,
            0,
            0
          );

          bicudo::gpu_memory_fetch(
            &pipeline,
            0,
            0,
            0,
            bicudo::types::WRITEBACK
          );
        }
      },
      ekg::action::activity
    );

  ekg::slider<uint32_t>("meow-gpu", ekg::dock::fill)
    ->set_text_align(ekg::dock::left | ekg::dock::center)
    ->range<uint32_t>(0, 0, 0, 100)
    ->range<uint32_t>(0).u32.transfer_ownership(&number_host);

  ekg::vec3 background_color {0.081f, 0.088f, 0.075f};
  ekg::label("BG:", ekg::dock::next);
  ekg::slider<float>("bg-clear-color-ownership", ekg::dock::fill)
    ->range<float>(0, 0.0f, 0.0f, 1.0f, 2)
    ->range<float>(0).f32.transfer_ownership(&background_color.x)
    ->range<float>(1, 0.0f, 0.0f, 1.0f, 2)
    ->range<float>(1).f32.transfer_ownership(&background_color.y)
    ->range<float>(2, 0.0f, 0.0f, 1.0f, 2)
    ->range<float>(2).f32.transfer_ownership(&background_color.z)
    ->set_text_align(ekg::dock::center | ekg::dock::right);

  ekg::label("DT:", ekg::dock::next);
  ekg::slider<float>("dt-ownership", ekg::dock::fill)
    ->range<float>(0, 0.0f, 0.0f, 1.0f, 5)
    ->range<float>(0).f32.transfer_ownership(&bicudo::dt)
    ->set_text_align(ekg::dock::center | ekg::dock::right);

  ekg::label("FPS:", ekg::dock::next);
  ekg::slider<uint64_t>("framerate-ownership", ekg::dock::fill)
    ->range<uint64_t>(0, 0, 0, 1000)
    ->range<uint64_t>(0).u64.transfer_ownership(&bicudo::current_framerate)
    ->set_text_align(ekg::dock::center | ekg::dock::right);

  ekg::label("Gravity:", ekg::dock::next);
  ekg::slider<float>("gravity-ownership", ekg::dock::fill)
    ->range<float>(0, 9.0f, 0.0f, 20.0f)
    ->range<float>(0).f32.transfer_ownership(&bicudo::app.world_manager.gravity.y)
    ->set_text_align(ekg::dock::center | ekg::dock::right);

  ekg::label("Objects:", ekg::dock::next);
  ekg::slider<uint64_t>("objects-ownership", ekg::dock::fill)
    ->range<uint64_t>(0, 0, 0, 10000)
    ->range<uint64_t>(0).u64.transfer_ownership(&meow::app.rendering_placements_count)
    ->set_text_align(ekg::dock::center | ekg::dock::left);

  ekg::label("---", ekg::dock::next);
  ekg::checkbox("GPU-ROCm Collisions", false, ekg::dock::next | ekg::dock::fill)
    ->transfer_ownership((bool*)&bicudo::app.physics_runtime_type);

  ekg::checkbox("Show AABB", false, ekg::dock::next | ekg::dock::fill)
    ->transfer_ownership(&meow::app.settings.show_aabb);

  ekg::checkbox("Show Collision Info", false, ekg::dock::next | ekg::dock::fill)
    ->transfer_ownership(&meow::app.settings.show_collision_info);

  ekg::checkbox("Show Vertices", false, ekg::dock::next | ekg::dock::fill)
    ->transfer_ownership(&meow::app.settings.show_vertices);

  ekg::ui::textbox *p_terminal {
    ekg::textbox("terminal", "\0", ekg::dock::fill | ekg::dock::next)
      ->set_scaled_height(8)
      ->set_typing_state(ekg::state::disable)
  };

  ekg::ui::label *p_position {ekg::label("", ekg::dock::next | ekg::dock::fill)};

  ekg::scrollbar("scrollbar-meow");
  ekg::pop_group();

  bicudo::app.world_manager.gravity.x = 0.0f;

  uint64_t framerate_count {};
  ekg::timing elapsed_frame_timing {};
  bicudo::vec2 gravity {};

  bicudo::object *p_cow {new bicudo::object({
    .p_tag = "vakinha",
    .mass = 2000.0f,
    .friction = 0.0001f,
    .restitution = 0.2f,
    .pos = {20, 20},
    .size = {144, 144},
    .acc = gravity
  })};

  bicudo::object *p_cow_2 {new bicudo::object({
    .p_tag = "gatinho",
    .mass = 20.0f,
    .friction = 0.0001f,
    .restitution = 0.2f,
    .pos = {200, 20},
    .size = {400, 50},
    .acc = gravity
  })};

  bicudo::object *p_terrain_bottom {new bicudo::object({
    .p_tag = "terrain-bottom",
    .mass = 0.0f,
    .friction = 0.8f,
    .restitution = 0.2f,
    .inertia = 0.0f,
    .pos = {200, 800},
    .size = {1280, 50},
    .acc = {0.0f, 0.0f}
  })};

  bicudo::object *p_terrain_top {new bicudo::object({
    .p_tag = "terrain-top",
    .mass = 0.0f,
    .friction = 0.2f,
    .restitution = 0.2f,
    .inertia = 0.0f,
    .pos = {200, 200},
    .size = {1280, 50},
    .acc = {0.0f, 0.0f}
  })};

  bicudo::object *p_terrain_left {new bicudo::object({
    .p_tag = "terrain-left",
    .mass = 0.0f,
    .friction = 0.2f,
    .restitution = 0.2f,
    .inertia = 0.0f,
    .pos = {200, 200},
    .size = {50, 1280},
    .acc = {0.0f, 0.0f}
  })};

  bicudo::object *p_terrain_right {new bicudo::object({
    .p_tag = "terrain-right",
    .mass = 0.0f,
    .friction = 0.2f,
    .restitution = 0.2f,
    .inertia = 0.0f,
    .pos = {900, 200},
    .size = {50, 1280},
    .acc = {0.0f, 0.0f}
  })};

  bicudo::object *p_picked_obj {nullptr};
  ekg::vec4 &interact {ekg::input::interact()};
  bicudo::vec2 drag {};

  bicudo::app.physics_runtime_type = bicudo::physics_runtime_type::GPU_ROCM;

  bicudo::init();
  bicudo::world::insert(p_cow);
  bicudo::world::insert(p_cow_2);
  /*bicudo::world::insert(p_terrain_top);
  bicudo::world::insert(p_terrain_bottom);
  bicudo::world::insert(p_terrain_left);
  bicudo::world::insert(p_terrain_right);*/

  for (uint64_t it {}; it < 0; it++) {
    bicudo::world::insert(new bicudo::object({
    .p_tag = "miau",
    .mass = bicudo_clamp_min(static_cast<float>(std::rand() % 200), 1),
    .friction = bicudo_clamp_min(static_cast<float>((std::rand() % 100) / 100), 0.0000001f),
    .restitution = bicudo_clamp_min(static_cast<float>((std::rand() % 100) / 100), 0.0000001f),
    .pos = {static_cast<float>(std::rand() % 800), static_cast<float>(std::rand() % 100)},
    .size = {bicudo_clamp_min(static_cast<float>(std::rand() % 200), 10.0f), bicudo_clamp_min(static_cast<float>(std::rand() % 200), 10.0f)},
    .acc = gravity
    }));
  }

  meow::init();

  ekg::ui::auto_scale = false;
  ekg::ui::scale = {1280.0f, 700.0f};

  while (running) {
    while (SDL_PollEvent(&sdl_event)) {
      if (sdl_event.type == SDL_QUIT) {
        running = false;
      }

      if (sdl_event.type == SDL_WINDOWEVENT && sdl_event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        bicudo::viewport(sdl_event.window.data1, sdl_event.window.data2);
        meow::app.immediate.set_viewport(sdl_event.window.data1, sdl_event.window.data2);
      }

      ekg::os::sdl_poll_event(sdl_event);

      meow::tools_pick_camera(
        &meow::app.camera_pickup_info
      );

      meow::tools_pick_object_from_world(
        &meow::app.object_pickup_info
      );
    }

    ekg::ui::dt = 0.016f;
    bicudo::dt = ekg::ui::dt;

    if (ekg::reach(elapsed_frame_timing, 1000) && ekg::reset(elapsed_frame_timing)) {
      bicudo::current_framerate = framerate_count;
      framerate_count = 0;

      std::string position {"("};
      position += std::to_string(bicudo::app.world_manager.camera.placement.pos.x);
      position += ", ";
      position += std::to_string(bicudo::app.world_manager.camera.placement.pos.y);
      position += ")";

      p_position->set_value(position);
    }

    meow::tools_update_picked_camera(
      &meow::app.camera_pickup_info
    );

    meow::tools_update_picked_object(
      &meow::app.object_pickup_info
    );

    bicudo::update();
    ekg::update();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(background_color.x, background_color.y, background_color.z, 1.0f);
    glViewport(0.0f, 0.0f, ekg::ui::width, ekg::ui::height);

    meow::render();
    ekg::render();

    framerate_count++;

    if (bicudo::log::buffered) {
      if (p_terminal->p_value->size() >= 100000) {
        p_terminal->p_value->erase(
          p_terminal->p_value->begin(),
          p_terminal->p_value->end() - 10000
        );
      }

      ekg::utf_decode(bicudo::log::buffer.str(), p_terminal->get_value());
      bicudo::log::flush();
    }

    SDL_GL_SwapWindow(p_sdl_win);
    SDL_Delay(bicudo::cpu_interval_ms);
  }

  return 0;
}