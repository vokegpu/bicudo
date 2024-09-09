# Bicudo

Bicudo is an useless physics engine being develop to process SAT and Newton's laws under GPU via HIP / ROCm.

# Building

You need all these libraries:

[EKG GUI Library](https://github.com/vokegpu/ekg-ui-library)
[Freetype](http://freetype.org/)
[SDL2](https://www.libsdl.org/)
[HIP/ROCm](https://github.com/ROCm/HIP)
[GLEW](https://glew.sourceforge.net/)

And a GCC/Clang compiler under Linux only (Windows port not yet).
Run the following command:

```sh
cmake -S . -B ./cmake-build-debug/ -G Ninja && cmake --build ./cmake-build-debug/
```

Outputs:
`/bin/linux/bicudoclient` (the starter client)
`/lib/linux/libbicudo.a` (the engine native library)
