#include <iostream>

#include "bicudo/bicudo.hpp"

int32_t main(int32_t, char**) {
  std::cout << "mu" << std::endl;
  bicudo::init();
  return 0;
}