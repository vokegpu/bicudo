#ifndef BICUDO_PIPELINE_HPP
#define BICUDO_PIPELINE_HPP

#include <bicudo/log/log.hpp>

namespace bicudo {
  enum pipeline_device_set_order : bicudo::device_id_t {
    FIRST_ONE = 0,
    SPECIFIC = 1
  };
}

namespace bicudo::pipeline {
  class base {
  public:
    base() {};
  public:
    virtual bicudo::result_t init() { return bicudo::result::NOT_IMPLEMENTED; };
  };
}

#endif
