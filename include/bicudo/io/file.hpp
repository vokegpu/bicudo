#ifndef BICUDO_IO_FILE_HPP
#define BICUDO_IO_FILE_HPP

#include <bicudo/log/log.hpp>

#include <ostream>
#include <string>

namespace bicudo {
  template<typename t>
  struct file_t {
  public:
    std::string tag {};
    std::string path {};    
    t content {};
  };

  template<typename t>
  using file_read_properties_t = file_t<t>;
}

namespace bicudo {
  bicudo::result_t read(
    bicudo::file_t<std::string> &file,
    bicudo::file_read_properties_t<std::string> &file_read_properties
  );
}

#endif
