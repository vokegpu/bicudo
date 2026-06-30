#!/bin/bash

# made by sabrina w.

runtest=0
build=0
install=0
amd=0

clear

echo "meow running~"

for arg in "$@"
do
  if [ $arg = "--build" ] || [ $arg = "-b" ]; then
    build=1

  fi

  if [ $arg = "--install" ] || [ $arg = "-i" ]; then
    install=1
  fi

  if [ $arg = "--test" ] || [ $arg = "-t" ]; then
    runtest=1
  fi

  if [ $arg = "--hip-rocm" ] || [ $arg = "-hr" ]; then
    amd=1
  fi
done

if [ $build = 1 ]; then
  cmake -S . -B ./cmake-build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=1 -DBICUDO_HIP_ROCM=$amd
  cmake --build ./cmake-build
fi

if [ $install = 1 ]; then
  cmake --install ./cmake-build
fi

if [ $runtest = 1 ]; then
  cd ./test/ && cmake -S . -B ./cmake-build -G Ninja -DCMAKE_EXPORT_COMPILE_COMMANDS=1 && cmake --build ./cmake-build && cd ./bin/ && ./bicudo-tests
fi

echo "meow 52~"
