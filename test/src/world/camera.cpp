#include "meow.hpp"
#include "world/camera.hpp"

void meow::camera::create() {
  
}

void meow::camera::set_zoom(float zoom) {
  this->interpolated_zoom = zoom;
}

void meow::camera::on_update() {
  this->rect.velocity += this->rect.acceleration * bicudo::dt;
  this->rect.velocity *= this->zoom;
  this->rect.velocity = bicudo::lerp(this->rect.velocity, 0.0f, 0.1f);
  this->rect.pos += this->rect.velocity;

  this->zoom = bicudo::lerp(this->zoom, this->interpolated_zoom, 0.3f);

  this->is_while_zoom = (this->interpolated_zoom != this->zoom);
  this->rect.mass = 0.0f;

  this->view.x = this->rect.pos.x;
  this->view.y = this->rect.pos.y;
  this->view.z = meow::app.immediate.viewport.z / meow::app.camera.zoom;
  this->view.w = meow::app.immediate.viewport.w / meow::app.camera.zoom;
}
