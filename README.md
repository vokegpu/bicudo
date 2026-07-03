## bicudo~

This is an useless 2D physics engine to be used with ROCm or CUDA via HIP, soon should have support for Intel, OpenCL, OpenGL4, and Vulkan. But now it is focused to HPC GPGPU-APIs.

The project is simple, but not done yet, so, wait for new commits and updates. By the 52.

### Installation

The `bicudo` library has only these dependencies: `HIP/ROCm`.

For AMD ROCm installation check official guides.

#### Linux

The development is under Arch Linux, make sure you know how install dependencies by yourself.

A file `bicudo-linux.sh` was made to help with development and instalation process. Make sure you `chmode u+x ./bicudo-linux.sh` before running.

For building process you must pass argument `--build` and complete with your desired API model implementation: `--hip-rocm`.

For example:
```
./bicudo-linux.sh --build --hip-rocm
sudo ./bicudo-linux.sh --install
```

`--install` argument installs on your usr local directory.

### Test

#### Linux

On Linux you can just pass `--test` to check if all is right. Remember to complete with your desired API model, if no API model is passed, then the CPU-model is used by default.

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

As shown, there is `Bicudo::bicudo-hip-rocm` and later others implementations. This is required for cross-multi-platform support. When using this library on your project, make sure you add to the installer of your game/software the properly GPU-API implementation.

If no implementation is inserted you wont be able to use the GPU-acceleration, only CPU-acceleration with basic SAT implementation for physics body.
