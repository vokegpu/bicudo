#include <bicudo/cpu/model.hpp>
 
bicudo::result_t bicudo::cpu::init() {
  bicudo::log("No accelerated devices, using central processor unit (CPU).");
  bicudo::logw("This is not software accelerated.");
  bicudo::log("Initialized with success.");

  return bicudo::result::SUCCESS;
}

bicudo::hypergroup_t &bicudo::cpu::new_hypergroup() {
  return *(this->hypergroups.emplace_back() = new bicudo::hypergroup_t { .unique_id = this->infspirit++ });
}

bicudo::body_t &bicudo::cpu::new_body(bicudo::hypergroup_t &hypergroup) {
  static bicudo::body_t not_found_body {};
  if (hypergroup != bicudo::found) {
    bicudo::loge("Invalid hypergroup, hypergroup must be property generated!");
    return not_found_body;
  }

  return *(hypergroup.bodies.emplace_back() = new bicudo::body_t { .unique_id = this->infspirit++ });
}
