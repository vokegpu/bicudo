#include <bicudo/physics/physics.hpp>
#include <bicudo/bicudo.hpp>

bicudo::hypergroup_t &bicudo::as_new_hypergroup() {
  return bicudo::p_core->p_base->new_hypergroup();
}

bicudo::body_t &bicudo::as_new_body(bicudo::hypergroup_t &hypergroup) {
  return bicudo::p_core->p_base->new_body(hypergroup);
}

void bicudo::update(bicudo::hypergroup_t &hypergroup) {

}

bool bicudo::detect(bicudo::body_t &b1, bicudo::body_t &b2) {
  for (bicudo::vec2_t<float> &e1 : b1.edges) {
    bicudo::vec2_t<float> dir = (b1.pos - e1).normalize();

    float amin {9999.0f}; float amax {};
    for (bicudo::vec2_t<float> &v1 : b1.vertices) {
      float plane = v1.dot(dir);
      if (plane < amin) amin = plane;
      if (plane > amax) amax = plane;
    }

    float bmin {9999.0f}; float bmax {};
      for (bicudo::vec2_t<float> &v2 : b2.vertices) {
      float plane = v2.dot(dir);       
      if (plane < bmin) bmin = plane;
      if (plane > bmax) bmax = plane;
    }

    bool l1 = amax < bmin && amin < bmax;
    bool l2 = bmax < amin && bmin < amax;
    if (l1 || l2) return false;
  }

  return true;
}

void bicudo::update(bicudo::body_t &body) {
  if (body.edges.empty()) {
    body.edges.resize(4);
  }

  if (body.vertices.empty()) {
    body.vertices.resize(4);
  }

  bicudo::vec2_t<float> &pos = body.pos;
  bicudo::vec2_t<float> &size = body.size;
  bicudo::vec4_t<float> &rect = body.rect;

  bicudo::vec2_t<float> &up = body.edges.at(0);
  bicudo::vec2_t<float> &left = body.edges.at(1);
  bicudo::vec2_t<float> &down = body.edges.at(2);
  bicudo::vec2_t<float> &right = body.edges.at(3);

  float midw = size.x / 2;
  float midh = size.y / 2;

  body.vertices.at(0) = bicudo::vec2_t<float>(pos.x - midw, pos.y - midh).rotate(body.angle, pos);
  body.vertices.at(1) = bicudo::vec2_t<float>(pos.x + midw, pos.y - midh).rotate(body.angle, pos);
  body.vertices.at(2) = bicudo::vec2_t<float>(pos.x + midw, pos.y + midh).rotate(body.angle, pos);
  body.vertices.at(3) = bicudo::vec2_t<float>(pos.x - midw, pos.y + midh).rotate(body.angle, pos);

  up = bicudo::vec2_t<float>(pos.x, pos.y - midh).rotate(body.angle, pos);
  left = bicudo::vec2_t<float>(pos.x + midw, pos.y).rotate(body.angle, pos);
  down = bicudo::vec2_t<float>(pos.x, pos.y + midh).rotate(body.angle, pos);
  right = bicudo::vec2_t<float>(pos.x - midw, pos.y).rotate(body.angle, pos);

  rect.x = pos.x - midw;
  rect.y = pos.y - midh;
  rect.z = size.x;
  rect.w = size.y;
}
