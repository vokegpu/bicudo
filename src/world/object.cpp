#include "bicudo/world/object.hpp"
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
  this->placement.min.x = 99999.0f;
  this->placement.min.y = 99999.0f;
  this->placement.max.x = -99999.0f;
  this->placement.max.y = -99999.0f;

  this->placement.velocity += this->placement.acc * bicudo::dt;
  this->placement.pos += this->placement.velocity;

  this->placement.angular_velocity += this->placement.angular_acc * bicudo::dt;
  this->placement.angle += this->placement.angular_velocity;

  bicudo::vec2 center {
    this->placement.pos.x + (this->placement.size.x / 2),
    this->placement.pos.y + (this->placement.size.y / 2)
  };

  for (bicudo::vec2 &vertex : this->placement.vertices) {
    vertex += this->placement.velocity;
    vertex = vertex.rotate(this->placement.angular_velocity, center);

    this->placement.min.x = std::min(this->placement.min.x, vertex.x);
    this->placement.min.y = std::min(this->placement.min.y, vertex.y);
    this->placement.max.x = std::max(this->placement.max.x, vertex.x);
    this->placement.max.y = std::max(this->placement.max.y, vertex.y);
  }

  bicudo::splash_edges_normalized(
    this->placement.edges.data(),
    this->placement.vertices.data()
  );
}