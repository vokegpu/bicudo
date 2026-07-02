#ifndef BICUDO_PIPELINE_HPP
#define BICUDO_PIPELINE_HPP

#include <bicudo/log/log.hpp>
#include <bicudo/physics/hypergroup.hpp>
#include <bicudo/physics/physics.hpp>

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
    virtual bicudo::result_t registry_hypergroup(bicudo::hypergroup_t *p_hypergroup) {return bicudo::result::NOT_IMPLEMENTED; };
    virtual bicudo::result_t registry_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) {return bicudo::result::NOT_IMPLEMENTED; };
    virtual bicudo::result_t unregistry_hypergroup(bicudo::hypergroup_t *p_hypergroup) {return bicudo::result::NOT_IMPLEMENTED;}
    virtual bicudo::result_t unregistry_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) {return bicudo::result::NOT_IMPLEMENTED;}
    virtual bicudo::result_t update_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) {return bicudo::result::NOT_IMPLEMENTED; };
    virtual bicudo::result_t update(bicudo::physics_update_mode mode) {return bicudo::result::NOT_IMPLEMENTED; };
  };
}

#endif
