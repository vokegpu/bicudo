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
    bicudo::hypergroup_t &new_hypergroup() override;
    bicudo::body_t &new_body(bicudo::hypergroup_t &hypergroup) override;
  };
}

#endif
