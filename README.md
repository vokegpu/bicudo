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

#### Physics Body and Hypergroups

not yet to show.

#### Techniques, and Physics

The project for CPU implementation uses SAT (separation axis theorem). This was implemented over the study of [the book (Michael Tanaya, Huaming Chen, Jebediah Pavleas, Kelvin Sung)](https://www.amazon.com/Building-Game-Physics-Engine-JavaScript/dp/1484225821).

For GPU acceleration check #6.

## 52
