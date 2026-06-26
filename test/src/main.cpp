#include <cstdint>

#include <bicudo/bicudo.hpp>
#include <bicudo/pipeline/rocm.hpp>

int32_t main(int32_t, char**) {
  bicudo::init_core_t bicudo_init_core = {
    .p_base = bicudo::rocm()
  };

  bicudo::core_t core {};
  bicudo::init(bicudo_init_core, core);

  bicudo::flush();
  return 0;
}
