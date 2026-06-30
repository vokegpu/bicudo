#include <cstdint>

#include <bicudo/pipeline/rocm.hpp>
#include <bicudo/bicudo.hpp>

int32_t main(int32_t, char**) {
  bicudo::init_core_t bicudo_init_core = {
    .p_base = bicudo::as_rocm()
  };

  bicudo::core_t core {};
  bicudo::init(bicudo_init_core, core);

  return bicudo::flush();
}
