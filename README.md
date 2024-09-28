# Bicudo 🐦

Bicudo is a physics engine library being develop to process Separation Axis Theorem (SAT) and Newton's laws under GPU via an GPGPU API (ROCm/HIP, OpenCL, CUDA/HIP). The project contains an optionally client called Meow OpenGL-4 based to test the library.

The name "Bicudo" refers to a baby chick I had some years ago, but 🐤 died... then I named this project as Bicudo! 💕

---

# Getting Started 😊

Bicudo library is a multi-API GPU-accelerated physics engine, bellow the supported APIs but you are able to use CPU-side too.

| API | Support | OS |
| --- | --- | --- |
| [ROCm/HIP](https://github.com/ROCm/HIP) | Buildable, Bad Performance | Windows & Linux |
| [OpenCL](https://github.com/KhronosGroup/OpenCL-SDK) | ? | ? |
| [CUDA/HIP](https://github.com/ROCm/HIP) | ? | ? |

ROCm is the priority, however it is only supported by RX series 6000 or higher. OpenCL must be implemented when the ROCm implementation be totally stable, safe and high-performance.

### Linker 🐈

Bicudo must be linked after all libraries, for example `-letc -lamdgpu64 -lbicudo`.

### Application-Implementation 🐈‍⬛

Bicudo is hardware-interface based, which means that you need by application initialize the API you want to use.

For example CPU-side only.
```C++
#include <bicudo/bicudo.hpp>

bicudo::runtime bicudo_runtime {
  .gravity = {}, // by default is 0.0 (x and y)
  .physics_runtime_type = bicudo::physics_runtime_type::CPU_SIDE
};

bicudo::init(&bicudo_runtime);
```

For example ROCm/HIP API:
```C++
#include <bicudo/bicudo.hpp>
#include <bicudo/api/rocm.hpp>

bicudo::runtime bicudo_runtime {
  .gravity = {}, // by default is 0.0 (x and y)
  .physics_runtime_type = bicudo::physics_runtime_type::GPU_ROCM,
  .p_rocm_api = new bicudo::api::rocm()
};

bicudo::init(&bicudo_runtime);
```

Running Bicudo is simple, be sure the application is calculating correctly the delta time.

Calling `bicudo::update(bicudo::runtime *p_runtime)` update position and process collision resolution.  
```C++
// any framework media layer initialization
// bicudo initialization

while (mainloop) {
  bicudo::dt = 0.16f; // 60fps
  bicudo::update(&bicudo_runtime);
  bicudo::log::flush(); // flush log

  // etc render
}
```

Alternatively there are two separate functions to update each physic step:  
-- `bicudo::update_position(bicudo::runtime *p_runtime, bicudo::physics::placement *p_placement)`  
-- `bicudo::update_collision(bicudo::runtime *p_runtime)`

```C++
// any framework media layer initialization
// bicudo initialization

while (mainloop) {
  bicudo::dt = 0.16f; // 60fps

  for (bicudo::physics::placement *&p_placements : bicudo_runtime.placement_list) {
    bicudo::update_position(
      &bicudo_runtime,
      p_placements
    );
  }

  bicudo::update_collisions(
    &bicudo_runtime
  );

  bicudo::log::flush(); // flush log

  // etc render
}
```

Rendering is application-side, Bicudo does not provide any rendering engine, only geometry data. All meshes are rotated but it is not the most efficient method send it all to the GPU, so on render application must calculate rotation using an Euller rotation-matrix or some other way.

```C++
bicudo::vec4 rect {};
for (bicudo::physics::placement *&p_placements : bicudo_runtime.placement_list) {
  rect.x = p_placements->pos.x;
  rect.y = p_placements->pos.y;
  rect.z = p_placements->size.x;
  rect.w = p_placements->size.y;

  if (!bicudo::vec4_collide_with_vec4(rect, my_camera_rect)) {
    continue; // frustum clip
  }

  bicudo::mat4 mat4x4_rotate = bicudo::mat4(1.0f);

  if (!bicudo::assert_float(p_placement->angle, 0.0f)) {
    bicudo::vec2 center {
      rect.x + (rect.z / 2), rect.y + (rect.w / 2)
    };

    mat4x4_rotate = bicudo::translate(mat4x4_rotate, center);
    mat4x4_rotate = bicudo::rotate(mat4x4_rotate, {0.0f, 0.0f, 1.0f}, -p_placement->angle);
    mat4x4_rotate = bicudo::translate(mat4x4_rotate, -center);
  }

  // do render etc
}
```

### Placement 🐮

Insert placements objects by function `bicudo::insert(bicudo::physics::placement *p_placement)`, the object physic properties is application-side.

```C++
bicudo::insert(
  new bicudo::physics::placement {
    .p_tag = "cow",
    .mass = 2000.0f, // for infinity mass set to 0.0 (no gravity effect).
    .friction = 0.8f, // min 0.0 max 1.0
    .restitution = 1.0f, // min 0.0 max 1.0
    .inertia = 0.2f, // min 0.0 max 1.0
    .pos = {20, 20},
    .size = {96, 96}
  }
);
```

---

# Bicudo Building 🔧🐦

Bicudo library requires all the three APIs include dir (ROCm/HIP, OpenCL, CUDA/HIP). For now only [ROCm/HIP](https://github.com/ROCm/HIP).  
Install the ROCm-SDK, for Windows download the official AMD installer, for Linux run command from the package manager.

For CMake building on Linux:
```sh
cmake -S . -B ./cmake-build-debug -G Ninja
cmake --build ./cmake-build-debug
```

Windows:
```sh
cmake -S . -B ./cmake-build-debug -G Ninja -DHIP_PATH="/path-to-rocm-dir/AMD/ROCm/x.x/"
cmake --build ./cmake-build-debug
```

Outputs: `/lib/windows/libbicudo.a`, `/lib/linux/libbicudo.a`

# Meow Building 🔧🐱

Meow is the graphical application used to test and showcase the Bicudo engine. It is not necessary, you can skip if you want.

For building Meow you must download all these libraries:  
[EKG GUI Library](https://github.com/vokegpu/ekg-ui-library)  
[FreeType](http://freetype.org/)  
[SDL2](https://www.libsdl.org/)  
[ROCm/HIP](https://github.com/ROCm/HIP)  
[GLEW](https://glew.sourceforge.net/)  

And a GCC/Clang compiler.
Run the following command:

```sh
cd ./meow # Meow is a sub-folder in this project

cmake -S . -B ./cmake-build-debug/ -G Ninja
cmake --build ./cmake-build-debug/
```

Outputs: `./meow/bin/meow`, `./meow/bin/meow.exe`

# Thanks 💕

Michael Tanaya (Author), Huaming Chen (Author), Jebediah Pavleas (Author), Kelvin Sung (Author); of book [Building a 2D Game Physics Engine: Using HTML5 and JavaScript](https://www.amazon.com/Building-Game-Physics-Engine-JavaScript/dp/1484225821) 😊🐄


