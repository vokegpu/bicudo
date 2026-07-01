#ifndef BICUDO_PHYSICS_HPP
#define BICUDO_PHYSICS_HPP

#include <bicudo/physics/hypergroup.hpp>

namespace bicudo {
  bicudo::hypergroup_t &as_new_hypergroup();
  bicudo::body_t &as_new_body(bicudo::hypergroup_t &hypergroup);

  void update(bicudo::hypergroup_t &hypergroup);
  void update(bicudo::body_t &body);
  bool detect(bicudo::body_t &b1, bicudo::body_t &b2);
}

#endif
