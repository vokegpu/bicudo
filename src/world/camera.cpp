#include "bicudo/world/camera.hpp"

bicudo::vec2 bicudo::camera::display {};

void bicudo::camera::create() {
  this->placement.p_tag = "bicudo-system-camera";
}

void bicudo::camera::set_viewport(int32_t w, int32_t h) {
  bicudo::camera::display.x = static_cast<float>(w);
  bicudo::camera::display.y = static_cast<float>(h);
}

void bicudo::camera::on_update() {
  this->placement.velocity += this->placement.acc * bicudo::dt;
  this->placement.pos += this->placement.velocity;
  this->placement.velocity = bicudo::lerp(this->placement.velocity, 0.0f, smooth_amount + bicudo::dt);

  this->rect.x = this->placement.pos.x;
  this->rect.y = this->placement.pos.y;
  this->rect.z = bicudo::camera::display.x;
  this->rect.w = bicudo::camera::display.y;
}