#ifndef BICUDO_LOG_HPP
#define BICUDO_LOG_HPP

#include <cstdint>
#include <sstream>
#include <iostream>

#define BICUDO_LOG_STATUS_PREFIX "<bicudo-status> "
#define BICUDO_LOG_ERROR_PREFIX "<bicudo-error> "
#define BICUDO_LOG_WARNING_PREFIX "<bicudo-warning> "

namespace bicudo {
  using id_t = std::size_t;
  using result_t = std::size_t;

  enum result : result_t {
    OK,
    SUCCESS,
    FAILED_TO_INITIALIZE_BICUDO,
    NOT_IMPLEMENTED
  };

  extern std::ostringstream buffer;
  extern bool buffered;

  void logr();

  template<typename t_t, typename... args_t>
  void logr(t_t first, args_t... rest) {
    bicudo::buffer << (first);
    logr(rest...);
  }

  template<typename t_t, typename... args_t>
  void log(t_t first, args_t... rest) {
    bicudo::buffer << BICUDO_LOG_STATUS_PREFIX << first;
    logr(rest...);
  }

  template<typename t_t, typename... args_t>
  void logw(t_t first, args_t... rest) {
    bicudo::buffer << BICUDO_LOG_WARNING_PREFIX << first;
    logr(rest...);
  }

  template<typename t_t, typename... args_t>
  void loge(t_t first, args_t... rest) {
    bicudo::buffer << BICUDO_LOG_ERROR_PREFIX << first;
    logr(rest...);
  }

  void flush();
}

#endif
