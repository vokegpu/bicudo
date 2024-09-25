#ifndef BICUDO_UTIL_LOG_HPP
#define BICUDO_UTIL_LOG_HPP

#include <iostream>
#include <sstream>

namespace bicudo {
  struct log {
  public:
    static std::ostringstream buffer;
    static bool buffered;
  public:
    log() {
      bicudo::log::buffer << "[Bicudo] "; 
      bicudo::log::buffered = true;
    }

    ~log() {
      bicudo::log::buffer << '\n';
    }

    template<typename t>
    bicudo::log &operator<<(const t &content) {
      bicudo::log::buffer << content;
      return *this;
    }

    static void flush() {
      if (bicudo::log::buffered) {
        std::cout << bicudo::log::buffer.str();
        bicudo::log::buffer = std::ostringstream {};
        bicudo::log::buffered = false;
      }
    }
  };
}

#define hiprtc_validate(result, warning) result != HIPRTC_SUCCESS  && &(bicudo::log() << "[GPU] " << warning)
#define hip_validate(result, warning) result != hipSuccess && &(bicudo::log() << "[GPU] " << hipGetErrorName(result) << ": " << warning)

#endif