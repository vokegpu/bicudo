#include <iostream>

#include "bicudo/gpu/model.hpp"
#include "bicudo/bicudo.hpp"

#include <ekg/ekg.hpp>
#include <ekg/os/ekg_opengl.hpp>
#include <ekg/os/ekg_sdl.hpp>

#include <GL/glew.h>
#include <SDL2/SDL.h>
#include "client.hpp"

client::application client::app {};

int32_t main(int32_t, char**) {
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

  ekg::frame("oiii muuu", {20, 20}, {400, 200})
    ->set_resize(ekg::dock::left | ekg::dock::bottom | ekg::dock::right | ekg::dock::top)
    ->set_drag(ekg::dock::full);

  ekg::button("couwnt in GPU:")
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

          bicudo::gpu_writeback(
            &pipeline,
            0,
            0,
            0
          );
        }
      },
      ekg::action::activity
    );

  ekg::slider<uint32_t>("meow-gpu", ekg::dock::fill)
    ->set_text_align(ekg::dock::left | ekg::dock::center)
    ->range<uint32_t>(0, 0, 0, 100)
    ->range<uint32_t>(0).u32.transfer_ownership(&number_host);

  ekg::vec3 background_color {0.34f, 0.03f, 0.9f};
  ekg::label("Background Color:", ekg::dock::next);
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

  ekg::pop_group();

  uint64_t framerate_count {};
  ekg::timing elapsed_frame_timing {};
  bicudo::vec2 gravity {};

  bicudo::object *p_cow {new bicudo::object({
    .p_tag = "vakinha",
    .mass = 20.0f,
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
    .size = {144, 144},
    .acc = gravity
  })};

  bicudo::object *p_terrain_bottom {new bicudo::object({
    .p_tag = "terrain-bottom",
    .mass = 0.0f,
    .friction = 0.2f,
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

  bicudo::init();
  bicudo::world::insert(p_cow);
  bicudo::world::insert(p_cow_2);
  bicudo::world::insert(p_terrain_top);
  bicudo::world::insert(p_terrain_bottom);
  bicudo::world::insert(p_terrain_left);
  bicudo::world::insert(p_terrain_right);

  ekg::input::bind("click-on-object", "mouse-1");
  ekg::input::bind("drop-object", "mouse-1-up");

  while (running) {
    while (SDL_PollEvent(&sdl_event)) {
      if (sdl_event.type == SDL_QUIT) {
        running = false;
      }

      if (sdl_event.type == SDL_WINDOWEVENT && sdl_event.window.event == SDL_WINDOWEVENT_SIZE_CHANGED) {
        bicudo::viewport(sdl_event.window.data1, sdl_event.window.data2);
      }

      ekg::os::sdl_poll_event(sdl_event);
      client::tools_pick_object_from_world(
        &client::app.pickup_info
      );
    }

    ekg::ui::dt = (1.0f / static_cast<float>(bicudo::current_framerate));
    bicudo::dt = ekg::ui::dt;

    if (ekg::reach(elapsed_frame_timing, 1000) && ekg::reset(elapsed_frame_timing)) {
      bicudo::current_framerate = framerate_count;
      framerate_count = 0;
    }

    client::tools_update_picked_object(
      &client::app.pickup_info
    );

    bicudo::update();
    ekg::update();

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glClearColor(background_color.x, background_color.y, background_color.z, 1.0f);
    glViewport(0.0f, 0.0f, ekg::ui::width, ekg::ui::height);

    bicudo::render();
    bicudo::log::flush();

    ekg::render();

    framerate_count++;

    SDL_GL_SwapWindow(p_sdl_win);
    SDL_Delay(bicudo::cpu_interval_ms);
  }

  return 0;
}