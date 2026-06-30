#ifndef BICUDO_CPU_MODEL_HPP
#define BICUDO_CPU_MODEL_HPP

#include <bicudo/pipeline/base.hpp>

namespace bicudo {
  class cpu : public bicudo::pipeline::base {
  public:
    cpu() : base() {};
  public:
    bicudo::result_t init() override;
  };
}

#endif
