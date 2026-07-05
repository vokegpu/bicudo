#include <bicudo/core/core.hpp>

bicudo::core_t *bicudo::p_core {};

std::string bicudo::device() {
  return bicudo::p_core->p_base->get_device_name();;
}
