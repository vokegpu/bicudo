# Bicudo

Bicudo is a physics engine library being develop to process SAT and Newton's laws under GPU via ROCm/HIP (or CPU-side only). The project uses a client called Meow OpenGL-4 based to test the library, but you are able to build the library only and uses on your application. It only requires ROCm/HIP.

# Bicudo Building

Bicudo library requires only [ROCm/HIP](https://github.com/ROCm/HIP) libary.

```sh
cmake -S . -B ./cmake-build-debug -G Ninja && cmake --build ./cmake-build-debug
```

Outputs: `/lib/windows/libbicudo.a`, `/lib/linux/libbicudo.a`

# Meow Building

Meow is the graphical application used to test and showcase the Bicudo engine. But it is not necessary, you can skip if you want.

For building Meow you must download all these libraries:  
[EKG GUI Library](https://github.com/vokegpu/ekg-ui-library)  
[FreeType](http://freetype.org/)  
[SDL2](https://www.libsdl.org/)  
[ROCm/HIP](https://github.com/ROCm/HIP)  
[GLEW](https://glew.sourceforge.net/)  

And a GCC/Clang compiler under Linux.
Run the following command:

```sh
cd ./meow # Meow is a sub-folder in this project
cmake -S . -B ./cmake-build-debug/ -G Ninja && cmake --build ./cmake-build-debug/
```

Outputs: `./meow/bin/meow`, `./meow/bin/meow.exe`

# Thanks

Michael Tanaya (Author), Huaming Chen (Author), Jebediah Pavleas (Author), Kelvin Sung (Author); of book [Building a 2D Game Physics Engine: Using HTML5 and JavaScript](https://www.amazon.com/Building-Game-Physics-Engine-JavaScript/dp/1484225821) 😊🐄


