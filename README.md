## bicudo~

This is a useless 2D physics engine to be used with ROCm or CUDA via HIP, soon should have support for Intel, OpenCL, OpenGL4, and Vulkan. But now it is focused to HPC GPGPU-APIs.

The project is simple, but not done yet, so, wait for new commits and updates. By the 52. 

### Setuping, building etc.

Building and installing:
```
sudo chmod +x ./bicudo-linux.sh
./bicudo-linux --build --install --hip-rocm
```

To be sure if it is running property, you can run tests:
```
./bicudo-linux .. --test
```

Soon should have `--hip-cuda`, `--opencl`, `--vulkan`, `--opengl4`.
