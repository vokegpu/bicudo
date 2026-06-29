#ifndef BICUDO_LOG_HPP
#define BICUDO_LOG_HPP

#include <cstdint>
#include <sstream>
#include <iostream>

#define BICUDO_LOG_STATUS_PREFIX  "< bicudo ok--- > "
#define BICUDO_LOG_ERROR_PREFIX   "< bicudo error > "
#define BICUDO_LOG_WARNING_PREFIX "< bicudo warn- > "

namespace bicudo {
  using id_t = std::size_t;
  using result_t = std::size_t;
  using device_id_t = int;
  
  constexpr std::size_t not_expected_memory_size {UINT64_MAX - 52};

  enum result : result_t {
    OK,
    SUCCESS,
    KERNEL_LOADED,
    KERNEL_NOT_LOADED,
    FAILED_TO_OPEN_FILE,
    FAILED_TO_COMPILE_KERNEL,
    FAILED_TO_INITIALIZE_BICUDO,
    FAILED_TO_INITIALIZE_ROCM,
    NOT_IMPLEMENTED
  };

  extern std::ostringstream buffer;
  extern bool buffered;
  extern int32_t exit_status;

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

  template<typename t>
  std::string logt(t &k, std::string_view type) {
    std::string l {};

    l += "<'";
    l += k.tag;
    l += "'-";
    l += type;
    l += "> ";

    return l;
  }

  template<typename t>
  std::string logtk(t &k) {
    return logt<t>(k, "kernel");
  }

  template<typename t>
  std::string logtf(t &k) {
    return logt<t>(k,"function");
  }

  template<typename t>
  std::string logtp(t &k) {
    return logt<t>(k, "pipeline");
  }

  int32_t flush();
}

#define bicudo_trace_log(x) std::cout << x << std::endl;
#define bicudo_hip_assert(result, expected, log) if (result != expected) log;

#endif
