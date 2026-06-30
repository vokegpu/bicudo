#include <cstdint>
#include <bicudo/pipeline/rocm.hpp>

#include "meow.hpp"

meow::application_t meow::app {}; 

int32_t main(int32_t, char**) {
  bicudo::init_core_t bicudo_init_core = {
    .p_base = bicudo::as_rocm()
  };

  bicudo::init(bicudo_init_core, meow::app.bicudo);

  return bicudo::flush();
}
