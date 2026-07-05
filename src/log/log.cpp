#include <bicudo/log/log.hpp>
#include <iostream>

std::ostringstream bicudo::buffer {};
bool bicudo::buffered {};
int32_t bicudo::exit_status {52};

void bicudo::logr() {
  bicudo::buffer << '\n';
  bicudo::buffered = true;
  bicudo::flush();
}

int32_t bicudo::flush() {
  if (bicudo::buffered) {
    std::cout << bicudo::buffer.str() << std::flush;
    bicudo::buffer = {};
    bicudo::buffered = false;
  }

  return bicudo::exit_status;
}
