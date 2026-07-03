#ifndef BICUDO_PHYSICS_HYPERGROUP_HPP
#define BICUDO_PHYSICS_HYPERGROUP_HPP

#include <bicudo/io/signature.hpp>
#include <bicudo/physics/body.hpp>
#include <vector>

namespace bicudo {
  struct hypergroup_t {
  public:
    std::vector<bicudo::body_t*> bodies {};
  public:
    bicudo_as_signed(bicudo::hypergroup_t);
  };
}

#endif
