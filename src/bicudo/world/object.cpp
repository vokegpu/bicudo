#include "object.hpp"
#include <iostream>

bicudo::object::object(bicudo::placement placement) {
  this->placement = placement;

  if (this->placement.mass > 0) {
    this->placement.mass = 1.0f / this->placement.mass;
  }

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
  bicudo::splash_vertices(
    this->placement.vertices.data(),
    this->placement.pos,
    this->placement.size
  );

  bicudo::splash_edges_normalized(
    this->placement.edges.data(),
    this->placement.vertices.data()
  );

  this->placement.velocity += this->placement.acc * bicudo::dt;
  bicudo::move(&this->placement, this->placement.velocity);
}