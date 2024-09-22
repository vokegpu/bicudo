#include "tools/camera.hpp"
#include "meow.hpp"

void meow::camera::create() {
  this->placement.p_tag = "bicudo-system-camera";
}

void meow::camera::on_update() {
  this->placement.velocity += this->placement.acc * bicudo::dt;
  this->placement.pos += this->placement.velocity;
  this->placement.velocity = bicudo::lerp(this->placement.velocity, 0.0f, smooth_amount + bicudo::dt);

  this->rect.x = this->placement.pos.x;
  this->rect.y = this->placement.pos.y;
  this->rect.z = meow::app.immediate.viewport.z / meow::app.camera.zoom;
  this->rect.w = meow::app.immediate.viewport.w / meow::app.camera.zoom;
}