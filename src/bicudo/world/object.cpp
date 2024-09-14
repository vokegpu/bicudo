#include "object.hpp"
#include <iostream>

bicudo::object::object(bicudo::placement placement) {
  this->placement = placement;
  bicudo::mass(&this->placement, placement.mass);

  this->placement.vertices.resize(4);
  bicudo::splash_vertices(
    this->placement.vertices.data(),
    this->placement.pos,
    this->placement.size
  );

  this->placement.edges.resize(4);
  bicudo::splash_edges_normalized(
    this->placement.edges.data(),
    this->placement.vertices.data()
  );
}

void bicudo::object::on_update() {
  this->placement.velocity += this->placement.acc * bicudo::dt;
  bicudo::move(&this->placement, this->placement.velocity);

  this->placement.angular_velocity += this->placement.angular_acc * bicudo::dt;
  bicudo::rotate(&this->placement, this->placement.angular_velocity);
}