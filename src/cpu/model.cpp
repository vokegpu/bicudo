#include <bicudo/cpu/model.hpp>
#include <bicudo/cpu/sat.hpp>
 
bicudo::result_t bicudo::cpu::init() {
  bicudo::log("No accelerated devices, using central processor unit (CPU).");
  bicudo::logw("This is not software accelerated.");
  bicudo::log("Initialized with success.");

  return bicudo::result::SUCCESS;
}

bicudo::result_t bicudo::cpu::registry_hypergroup(bicudo::hypergroup_t *p_hypergroup) {
  p_hypergroup->unique_id = this->infspirit++;
  this->hypergroups.push_back(p_hypergroup);
  return bicudo::result::OK;
}

bicudo::result_t bicudo::cpu::registry_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body
) {
  p_body->unique_id = this->infspirit++;
  p_hypergroup->bodies.push_back(p_body);
  return bicudo::result::OK;
}

bicudo::result_t bicudo::cpu::unregistry_hypergroup(
  bicudo::hypergroup_t *p_hypergroup
) {
  for (std::size_t i {}; i < this->hypergroups.size(); i++) {
    if (this->hypergroups.at(i) != p_hypergroup) continue;
    this->hypergroups.erase(this->hypergroups.begin() + i);
    return bicudo::result::SUCCESS;
  }

  return bicudo::result::FAILED;
}

bicudo::result_t bicudo::cpu::unregistry_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body) {

  for (bicudo::hypergroup_t *p_hp : this->hypergroups) {
    if (p_hp != p_hypergroup) continue;
    for (std::size_t i {}; i < p_hp->bodies.size(); i++) {
      if (p_hp->bodies.at(i) != p_body) continue;
      p_hp->bodies.erase(p_hp->bodies.begin() + i);
      return bicudo::result::SUCCESS;
    }
  }

  return bicudo::result::FAILED;
}

bicudo::result_t bicudo::cpu::update_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body
) {
  bicudo::cpu_sat_update_body(
    *p_body
  );

  return bicudo::result::OK;
}

bicudo::result_t bicudo::cpu::update(
  bicudo::physics_update_mode mode
) {

  bicudo::cpu_sat_collide_info_t info {};
  for (bicudo::hypergroup_t *p_hypergroup : this->hypergroups) {
    for (bicudo::body_t *p_body_a : p_hypergroup->bodies) {
      for (bicudo::body_t *p_body_b : p_hypergroup->bodies) {
        if (p_body_a == p_body_b) continue;
        info = bicudo::cpu_sat_collided(
          *p_body_a,
          *p_body_b
        );
      }
    }
  }

  return bicudo::result::OK;
}

bicudo::result_t bicudo::cpu::size_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body,
  bicudo::vec2_t<float> size
) {
  bicudo::cpu_sat_size(
    *p_body,
    size
  );

  return bicudo::result::OK;
}

bicudo::result_t bicudo::cpu::move_body(
  bicudo::hypergroup_t *p_hypergroup,
  bicudo::body_t *p_body,
  bicudo::vec2_t<float> direction
) {
  bicudo::cpu_sat_move(
    *p_body,
    direction
  );

  return bicudo::result::OK;
}
