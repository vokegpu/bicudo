#include <bicudo/physics/physics.hpp>
#include <bicudo/bicudo.hpp>

bicudo::result_t bicudo::registry(
  bicudo::hypergroup_t *p_hypergroup
) {
  return bicudo::p_core->p_base->registry_hypergroup(p_hypergroup);
}

bicudo::result_t bicudo::registry(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body
) {
  return bicudo::p_core->p_base->registry_body(p_hypergroup, p_body);
}

bicudo::result_t bicudo::unregistry(
  bicudo::hypergroup_t *p_hypergroup
) {
  return bicudo::p_core->p_base->unregistry_hypergroup(p_hypergroup);
}

bicudo::result_t bicudo::unregistry(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body
) {
  return bicudo::p_core->p_base->unregistry_body(p_hypergroup, p_body);
}

bicudo::result_t bicudo::update(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body
) {
  return bicudo::p_core->p_base->update_body(p_hypergroup, p_body);
}

bicudo::result_t bicudo::update(bicudo::physics_update_mode mode) {
  return bicudo::p_core->p_base->update(mode);
}
