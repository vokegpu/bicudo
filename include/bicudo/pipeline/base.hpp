#ifndef BICUDO_PIPELINE_HPP
#define BICUDO_PIPELINE_HPP

#include <bicudo/log/log.hpp>

namespace bicudo::pipeline {
  class base {
  public:
    virtual bicudo::result_t init() { return bicudo::result::NOT_IMPLEMENTED; };
  };
}

#endif
