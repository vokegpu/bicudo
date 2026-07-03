#include <bicudo/cpu/sat.hpp>

void bicudo::cpu_sat_solve(
  bicudo::cpu_sat_collide_info_t &info,
  bicudo::body_t &a,
  bicudo::body_t &b,
  float correction_rate
) {
  float amass {1.0f / a.mass};
  float bmass {1.0f / b.mass};

  a.inertia = 1.0f / ((1.0f / amass) * (a.size.magnitude() / 12.0f));
  b.inertia = 1.0f / ((1.0f / bmass) * (a.size.magnitude() / 12.0f));

  float magnitude = info.depth / (amass + bmass) * correction_rate;
  bicudo::vec2_t<float> correct_amount = info.dir * magnitude;

  bicudo::cpu_sat_move(a, correct_amount * -amass);
  bicudo::cpu_sat_move(b, correct_amount * bmass);

  bicudo::vec2_t<float> start = info.start * (bmass / (amass + bmass));
  bicudo::vec2_t<float> end = info.end * (bmass / (amass + bmass));
  bicudo::vec2_t<float> p = start + end;

  bicudo::vec2_t<float> r1 = p - a.pos;
  bicudo::vec2_t<float> r2 = p - b.pos;

  bicudo::vec2_t<float> v1 = a.velocity + bicudo::vec2_t<float>(-1 * a.angular_velocity * r1.y, a.angular_velocity * r1.x);
  bicudo::vec2_t<float> v2 = b.velocity + bicudo::vec2_t<float>(-1 * b.angular_velocity * r2.y, b.angular_velocity * r2.x);

  bicudo::vec2_t<float> rvel = v2 - v1;
  float rvelproj = rvel.dot(info.dir);

  if (rvelproj > 0.0f) return;

  float r1xn = r1.cross(info.dir);
  float r2xn = r2.cross(info.dir);

  float rest = std::min(a.restitution, b.restitution);
  float jn = (-(1 + rest) * rvelproj)
    / (amass + bmass + r1xn * r1xn * a.inertia + r2xn * r2xn * b.inertia);

  bicudo::vec2_t<float> impulse = info.dir * jn;

  a.velocity -= impulse * amass;
  b.velocity += impulse * bmass;

  a.angular_velocity -= r1xn * jn * a.inertia;
  b.angular_velocity += r2xn * jn * b.inertia;

  bicudo::vec2_t<float> tan = (rvel - (info.dir * rvelproj)).normalize() * -1.0f;

  float r1xt = r1.cross(tan);
  float r2xt = r2.cross(tan);

  float fric = std::min(a.friction, b.friction);
  float jt = (-(1 + rest) * rvel.dot(tan) * fric)
    / (amass + bmass + r1xt * r1xt * a.inertia + r2xt * r2xt * b.inertia);

  jt = jt > jn ? jn : jt;
  impulse = tan * jt;

  a.velocity -= impulse * amass;
  b.velocity += impulse * bmass;

  a.angular_velocity -= r1xt * jt * a.inertia;
  b.angular_velocity += r2xt * jt * b.inertia;
}

bicudo::cpu_sat_collide_info_t bicudo::cpu_sat_check_collide(
  bicudo::body_t &a,
  bicudo::body_t &b
) {
  float bestdist {99999.0f};
  float maxdist {};

  bicudo::vec2_t<float> point {};
  bicudo::vec2_t<float> bestpoint {};

  std::size_t best {};
  std::size_t edges_count {a.edges.size()};
  bool has {edges_count > 0};

  cpu_sat_collide_info_t info {};
  for (std::size_t i {}; has && i < edges_count; i++) {
    bicudo::vec2_t<float> normal = a.edges.at(i) * -1.0f;
    bicudo::vec2_t<float> v = a.vertices.at(i);

    maxdist = -99999.0f;
    has = false;

    for (bicudo::vec2_t<float> &bv : b.vertices) {
      bicudo::vec2_t<float> dir = bv - v;
      float proj = dir.dot(normal);

      if (proj > 0 && proj > maxdist) {
        maxdist = proj;
        point = bv;
        has = true;
      }
    }

    if (has && maxdist < bestdist) {
      bestdist = maxdist;
      best = i;
      bestpoint = point;
    }
  }

  if (has) {
    info.dir = a.edges.at(best);
    info.depth = bestdist;
    info.start = bestpoint + (info.dir * info.depth);
    info.end = info.start + (info.dir * info.depth);
  }

  info.collided = has;
  return info;
}

bicudo::cpu_sat_collide_info_t bicudo::cpu_sat_collided(
  bicudo::body_t &a,
  bicudo::body_t &b
) {
  bicudo::cpu_sat_collide_info_t info_a {};
  bicudo::cpu_sat_collide_info_t info_b {};
  bicudo::cpu_sat_collide_info_t info {};

  a.has_collide = info.collided;

  if (
      !(
        (info_a = bicudo::cpu_sat_check_collide(a, b))
        &&
        (info_b = bicudo::cpu_sat_check_collide(b, a))
      )
    ) {
    return info;
  }

  if (info_a.depth < info_b.depth) {
    info.dir = info_a.dir;
    info.depth = info_a.depth;
    info.start = info_a.start - (info_a.dir * info_a.depth);
    info.end = info.start + (info.dir * info.depth);
  } else {
    info.dir = info_b.dir * -1.0f;
    info.depth = info_b.depth;
    info.start = info_b.start;
    info.end = info_b.start + (info.dir * info.depth);
  }

  info.collided = true;
  a.has_collide = info.collided;

  return info;
}

void bicudo::cpu_sat_update_body(
  bicudo::body_t &body
) {
  if (body.edges.empty()) {
    body.edges.resize(4);
  }

  if (body.vertices.empty()) {
    body.vertices.resize(4);
  }

  body.min.x = 99999.0f;
  body.min.y = 99999.0f;
  body.max.x = -99999.0f;
  body.max.y = -99999.0f;

  body.velocity += body.acceleration * bicudo::dt;
  body.pos += body.velocity;

  body.angular_velocity += body.angle_acceleration * bicudo::dt;
  body.angle += body.angular_velocity;

  float midw = body.size.x / 2;
  float midh = body.size.y / 2;

  body.vertices.at(0) = bicudo::vec2_t<float>(body.pos.x - midw, body.pos.y - midh);
  body.vertices.at(1) = bicudo::vec2_t<float>(body.pos.x + midw, body.pos.y - midh);
  body.vertices.at(2) = bicudo::vec2_t<float>(body.pos.x + midw, body.pos.y + midh);
  body.vertices.at(3) = bicudo::vec2_t<float>(body.pos.x - midw, body.pos.y + midh);

  for (bicudo::vec2_t<float> &vertex : body.vertices) {
    vertex = vertex.rotate(body.angular_velocity, body.pos);

    body.min.x = std::min(body.min.x, vertex.x);
    body.min.y = std::min(body.min.y, vertex.y);
    body.max.x = std::max(body.max.x, vertex.x);
    body.max.y = std::max(body.max.y, vertex.y);    
  }

  bicudo::vec2_t<float> &up = body.edges.at(0);
  bicudo::vec2_t<float> &right = body.edges.at(1);
  bicudo::vec2_t<float> &down = body.edges.at(2);
  bicudo::vec2_t<float> &left = body.edges.at(3);

  up = (body.vertices.at(1) - body.vertices.at(2)).normalize();
  right = (body.vertices.at(2) - body.vertices.at(3)).normalize();
  down = (body.vertices.at(3) - body.vertices.at(0)).normalize();
  left = (body.vertices.at(0) - body.vertices.at(1)).normalize();

  body.rect.x = body.pos.x - midw;
  body.rect.y = body.pos.y - midh;
  body.rect.z = body.size.x;
  body.rect.w = body.size.y;
}

void bicudo::cpu_sat_size(
  bicudo::body_t &body,
  bicudo::vec2_t<float> size
) {
  if (body.edges.empty()) {
    body.edges.resize(4);
  }

  if (body.vertices.empty()) {
    body.vertices.resize(4);
  }

  float midw = size.x / 2;
  float midh = size.y / 2;

  body.vertices.at(0) = bicudo::vec2_t<float>(body.pos.x - midw, body.pos.y - midh);
  body.vertices.at(1) = bicudo::vec2_t<float>(body.pos.x + midw, body.pos.y - midh);
  body.vertices.at(2) = bicudo::vec2_t<float>(body.pos.x + midw, body.pos.y + midh);
  body.vertices.at(3) = bicudo::vec2_t<float>(body.pos.x - midw, body.pos.y + midh);

  bicudo::vec2_t<float> &up = body.edges.at(0);
  bicudo::vec2_t<float> &right = body.edges.at(1);
  bicudo::vec2_t<float> &down = body.edges.at(2);
  bicudo::vec2_t<float> &left = body.edges.at(3);

  up = (body.vertices.at(1) - body.vertices.at(2)).normalize();
  right = (body.vertices.at(2) - body.vertices.at(3)).normalize();
  down = (body.vertices.at(3) - body.vertices.at(0)).normalize();
  left = (body.vertices.at(0) - body.vertices.at(1)).normalize();
}

void bicudo::cpu_sat_move(
  bicudo::body_t &body,
  bicudo::vec2_t<float> direction
) {
  body.min.x = 99999.0f;
  body.min.y = 99999.0f;
  body.max.x = -99999.0f;
  body.max.y = -99999.0f;

  float midw = body.size.x / 2;
  float midh = body.size.y / 2;

  body.vertices.at(0) = bicudo::vec2_t<float>(body.pos.x - midw, body.pos.y - midh);
  body.vertices.at(1) = bicudo::vec2_t<float>(body.pos.x + midw, body.pos.y - midh);
  body.vertices.at(2) = bicudo::vec2_t<float>(body.pos.x + midw, body.pos.y + midh);
  body.vertices.at(3) = bicudo::vec2_t<float>(body.pos.x - midw, body.pos.y + midh);

  for (bicudo::vec2_t<float> &vertex : body.vertices) {
    vertex += direction;

    body.min.x = std::min(body.min.x, vertex.x);
    body.min.y = std::min(body.min.y, vertex.y);
    body.max.x = std::max(body.max.x, vertex.x);
    body.max.y = std::max(body.max.y, vertex.y);
  }

  bicudo::vec2_t<float> &up = body.edges.at(0);
  bicudo::vec2_t<float> &right = body.edges.at(1);
  bicudo::vec2_t<float> &down = body.edges.at(2);
  bicudo::vec2_t<float> &left = body.edges.at(3);

  up = (body.vertices.at(1) - body.vertices.at(2)).normalize();
  right = (body.vertices.at(2) - body.vertices.at(3)).normalize();
  down = (body.vertices.at(3) - body.vertices.at(0)).normalize();
  left = (body.vertices.at(0) - body.vertices.at(1)).normalize();

  body.pos += direction;
}
