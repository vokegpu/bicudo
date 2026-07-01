#ifndef BICUDO_PIPELINE_HPP
#define BICUDO_PIPELINE_HPP

#include <bicudo/log/log.hpp>
#include <bicudo/physics/hypergroup.hpp>

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
  public:
    virtual bicudo::hypergroup_t &new_hypergroup() { static bicudo::hypergroup_t not_found {}; return not_found; };
    virtual bicudo::body_t &new_body(bicudo::hypergroup_t &hypergroup) { static bicudo::body_t not_found {}; return not_found; };
  };
}

#endif
