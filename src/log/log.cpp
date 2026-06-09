#include <bicudo/log/log.hpp>
#include <iostream>

std::ostringstream bicudo::buffer {};
bool bicudo::buffered {};

void bicudo::logr() {
  bicudo::buffer << '\n';
  bicudo::buffered = true;
}

void bicudo::flush() {
  if (bicudo::buffered) {
    std::cout << bicudo::buffer.str() << std::flush;
    bicudo::buffer = {};
    bicudo::buffered = false;
  }
}
