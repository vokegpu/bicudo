## bicudo~

This is a useless 2D physics engine to be used with ROCm or CUDA via HIP, and soon should have support for Intel, OpenCL, OpenGL4, and Vulkan. But now it is focused on HPC GPGPU-APIs.

The project is simple but not done yet, so wait for new commits and updates.

52.

---

### Installation

The `bicudo` library has only these dependencies: `HIP/ROCm`.

For AMD ROCm installation, check the official guides.

#### Linux

The development is under Arch Linux; make sure you know how to install dependencies by yourself.

A file `bicudo-linux.sh` was made to help with the development and installation process. Make sure you `chmode u+x ./bicudo-linux.sh` before running.

For building, you must pass the argument `--build` and complete it with your desired API model implementation: `--hip-rocm`.

For example:
```
./bicudo-linux.sh --build --hip-rocm
sudo ./bicudo-linux.sh --install
```

`--install` argument installs in your usr local directory.

### Test

#### Linux

To test if everything is working properly, pass `--test`. Remember to complete with your desired API model; if no API model argument was passed, then the CPU model is used by default.

---

### Usage

The usage of bicudo is a little different due to multi-GPU support, but VokeGPU made it easy for you.

#### CMake

```CMake
find_package(Bicudo REQUIRED)

## ...

target_link_libraries(
  ...
  Bicudo::bicudo ## library
  Bicudo::bicudo-hip-rocm ## DLL/shared-library to ROCm support
  ...
)
```

As shown, there is `Bicudo::bicudo-hip-rocm` and later other implementations. This is required for cross-platform support. When using this library in your project, you must distribute the correct bicudo GPU-model implementation (for example, for AMD users, you should distribute the DLL/so `bicudo-hip-rocm.*`).

If no GPU model implementation is linked to the project, only CPU acceleration is supported.

#### Init

For initializing the bicudo properly you need to specificy the base you wish to use.

For AMD ROCm:
```cpp
#include <bicudo/pipeline/rocm.hpp>

bicudo::init_core_t bicudo_init_core = {
  .p_base = new bicudo::rocm()
};
```

For software-accelerated (CPU):
```cpp
#include <bicudo/pipeline/cpu.hpp>

bicudo::init_core_t bicudo_init_core = {
  .p_base = new bicudo::cpu()
};
```

Then init the core, you need to take care of `bicudo::core_t`.

```cpp
bicudo::core_t bicudo_core {};

bicudo::init(
  bicudo_init_core,
  bicudo_core
);
```

#### Physics Body and Hypergroups

Bicudo works with two concepts: `hypergroup_t` and `body_t`.

##### Hypergroup

Hypergroups are groups of `bicudo::body_t`, you can multiples context of bodies using hypergroups, this can helpful for some parallel colliding thing.

For registering:
```cpp
bicudo::hypergroup_t hypergroup {};
bicudo::registry(&hypergroup);
```

##### Body

Bodies are the moveable objects, where store position etc. I wont add documentation much for now, since it is not done yet. Just check the fields and use as how it works for you.

```cpp
bicudo::body_t my_body {};
bicudo::registry(&hypergroup, &my_body);
```

#### Running

You need to setup the delta-time by your self. Here hows I do.

```cpp
while (true) {
  bicudo::dt = 1.0f / last_frame_count;

  bicudo::update(
    bicudo::physics_update_mode::EVERYTHING
  );

  // ...
  // you also can update separetly by:
  bicudo::update(&hypergroup, &my_body);
    // this is useful for pickup
}
```

#### Techniques, and Physics

The project for CPU implementation uses SAT (separation axis theorem). This was implemented over the study of [the book (Michael Tanaya, Huaming Chen, Jebediah Pavleas, Kelvin Sung)](https://www.amazon.com/Building-Game-Physics-Engine-JavaScript/dp/1484225821).

For GPU acceleration, check [#7](https://github.com/vokegpu/bicudo/issues/7) discussion.

---

### Demo

This is the `meow` test; the point is physics, but art above all dead geometry.

https://github.com/user-attachments/assets/0c23dedd-1747-45cc-a0cb-fd9284afee9a

---

## 52~

'Eu vos invoco, pela minha inútil existência, não temerei uma bela batalha nem uma grande virtude de estar errada. Em fé em Deus, por Cristo e pelos 52.  
Soberano Humano, ponha-se sobre todos os tijolos das suas lamentações e dê forma, vida. O princípio criador, derivado em vosso Soberano Humano pela sua Alma criadora.'  
`- Sabrina W. 16:37:47 03/07/2026`
