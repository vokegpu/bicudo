#ifndef BICUDO_UTIL_LOG_HPP
#define BICUDO_UTIL_LOG_HPP

#include <iostream>
#include <ostream>

#include <hip/hiprtc.h>
#include <hip/hip_runtime.h>

namespace bicudo {
  struct log {
  public:
    static std::ostringstream buffer {};
    static bool buffered {};
  public:
    bicudo::log() {
      bicudo::log::buffer << "[Bicudo] "; 
    }

    ~bicudo::log() {
      bicudo::buffer << '\n';
    }

    template<typename t>
    bicudo::log &operator<<(t content) {
      bicudo::log::buffer << content << std::endl;
      return *this;
    }

    void flush() {
      if (bicudo::log::buffered) {
        std::cout << bicudo::buffer << std::endl;
        bicudo::buffer = {};
        bicudo::log::buffered = false;
      }
    }
  }
}

#define hiprtc_validate(result, warning) result != HIP_SUCCESS && (bicudo::log() << "[GPU] " << warning)
#define hip_validate(result, warning) result != hipSuccess && (bicudo::log() << "[GPU] " << warning)

#endif