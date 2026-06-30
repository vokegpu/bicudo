#include <bicudo/pipeline/cpu.hpp>
#include <bicudo/cpu/model.hpp>

bicudo::pipeline::base *bicudo::as_cpu() {
  return new bicudo::cpu();
}
