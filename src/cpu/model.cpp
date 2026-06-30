#include <bicudo/cpu/model.hpp>

bicudo::result_t bicudo::cpu::init() {
  bicudo::log("No accelerated devices, using central processor unit (CPU).");
  bicudo::logw("This is not software accelerated.");
  bicudo::log("Initialized with success.");

  return bicudo::result::SUCCESS;
}
