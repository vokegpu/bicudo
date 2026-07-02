#ifndef BICUDO_PHYSICS_HPP
#define BICUDO_PHYSICS_HPP

#include <bicudo/physics/hypergroup.hpp>

namespace bicudo {
  enum physics_update_mode {
    EVERYTHING,
    ONLY_COLLISION
  };

  bicudo::result_t registry(
    bicudo::hypergroup_t *p_hypergroup
  );

  bicudo::result_t registry(
    bicudo::hypergroup_t *p_hypergroup,
    bicudo::body_t *p_body
  );

  bicudo::result_t unregistry(
    bicudo::hypergroup_t *p_hypergroup
  );

  bicudo::result_t unregistry(
    bicudo::hypergroup_t *p_hypergroup,
    bicudo::body_t *p_body
  );

  bicudo::result_t update(
    bicudo::hypergroup_t *p_hypergroup,
    bicudo::body_t *p_body
  );

  bicudo::result_t update(
    bicudo::physics_update_mode mode
  );
}

#endif
