# Bicudo

Bicudo is a physics engine library being develop to process SAT and Newton's laws under GPU via HIP / ROCm (or CPU-side only). The project uses a client OpenGL 4 based to test the library, but you are able to build the library only and uses on your application. It only requires ROCm.

# Building

You need all these libraries:

[EKG GUI Library](https://github.com/vokegpu/ekg-ui-library)  
[FreeType](http://freetype.org/)  
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
