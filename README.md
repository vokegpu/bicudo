### bicudo~

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
