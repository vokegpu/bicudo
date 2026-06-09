#include <bicudo/pipeline/rocm.hpp>

bicudo::pipeline::base *bicudo::rocm() {
  return new bicudo::pipeline::rocm();
}
