#include "meow.hpp"
#include "world/camera.hpp"

void meow::camera::create() {
  
}

void meow::camera::on_update() {
  this->rect.velocity += this->rect.acceleration * bicudo::dt;
  this->rect.pos += this->rect.velocity;
  this->rect.velocity = bicudo::lerp(this->rect.velocity, 0.0f, smooth_amount + bicudo::dt);

  this->view.x = this->rect.pos.x;
  this->view.y = this->rect.pos.y;
  this->view.z = meow::app.immediate.viewport.z / meow::app.camera.zoom;
  this->view.w = meow::app.immediate.viewport.w / meow::app.camera.zoom;
}
