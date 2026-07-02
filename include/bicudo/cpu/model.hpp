#ifndef BICUDO_CPU_MODEL_HPP
#define BICUDO_CPU_MODEL_HPP

#include <bicudo/pipeline/base.hpp>

namespace bicudo {
  class cpu : public bicudo::pipeline::base {
  protected:
    bicudo::id_t infspirit {};
    std::vector<bicudo::hypergroup_t*> hypergroups {};
  public:
    cpu() : base() {};
  public:
    bicudo::result_t init() override;
    bicudo::result_t registry_hypergroup(bicudo::hypergroup_t *p_hypergroup) override;
    bicudo::result_t registry_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) override;
    bicudo::result_t unregistry_hypergroup(bicudo::hypergroup_t *p_hypergroup) override;
    bicudo::result_t unregistry_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) override;
    bicudo::result_t update_body(bicudo::hypergroup_t *p_hypergroup, bicudo::body_t *p_body) override;
    bicudo::result_t update(bicudo::physics_update_mode mode) override;
  };
}

#endif
