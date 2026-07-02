#ifndef BICUDO_MATH_GEOMETRY_HPP
#define BICUDO_MATH_GEOMETRY_HPP

#include <cfloat>
#include <cstdint>
#include <math.h>
#include <iostream>

#define bicudo_clamp_min(a, b) ((a) < (b) ? (b) : (a))
#define bicudo_clamp_max(a, b) ((a) > (b) ? (b) : (a))
#define bicudo_deg2rad(x) ((x) * 0.0174533f)

namespace bicudo {
  extern float dt;

  constexpr bool assert_float(float x, float y) {
    return fabsf(x - y) <= FLT_EPSILON * fmaxf(fabsf(x), fabsf(y));
  }

  template<typename t>
  struct mat4_t {
  public:
    union {
      struct {
        t _11 {}, _12 {}, _13 {}, _14 {};
        t _21 {}, _22 {}, _23 {}, _24 {};
        t _31 {}, _32 {}, _33 {}, _34 {};
        t _41 {}, _42 {}, _43 {}, _44 {};
      };

      t buffer[16];
    };

    inline t &operator[](std::size_t index) {
      return this->buffer[index];
    }

    t *data() {
      return this->buffer;
    }
  public:
    inline mat4_t(t identity = 1.0f) {
      this->_11 = this->_22 = this->_33 = this->_44 = identity;
    }

    inline mat4_t(
      t f11, t f12, t f13, t f14,
      t f21, t f22, t f23, t f24,
      t f31, t f32, t f33, t f34,
      t f41, t f42, t f43, t f44
    ) {
      this->_11 = f11; this->_12 = f12; this->_13 = f13; this->_14 = f14;
      this->_21 = f21; this->_22 = f22; this->_23 = f23; this->_24 = f24;
      this->_31 = f31; this->_32 = f32; this->_33 = f33; this->_34 = f34;
      this->_41 = f41; this->_42 = f42; this->_43 = f43; this->_44 = f44;
    }

    template<typename s>
    inline bicudo::mat4_t<t> operator*(bicudo::mat4_t<s> &r) {
      bicudo::mat4_t<t> &a {*this};
      bicudo::mat4_t<s> &b {r};
      bicudo::mat4_t<t> result {};

      result[0]  = a[0] * b[0] + a[4] * b[1] + a[8]  * b[2] + a[12] * b[3];
      result[1]  = a[1] * b[0] + a[5] * b[1] + a[9]  * b[2] + a[13] * b[3];
      result[2]  = a[2] * b[0] + a[6] * b[1] + a[10] * b[2] + a[14] * b[3];
      result[3]  = a[3] * b[0] + a[7] * b[1] + a[11] * b[2] + a[15] * b[3];

      result[4]  = a[0] * b[4] + a[4] * b[5] + a[8]  * b[6] + a[12] * b[7];
      result[5]  = a[1] * b[4] + a[5] * b[5] + a[9]  * b[6] + a[13] * b[7];
      result[6]  = a[2] * b[4] + a[6] * b[5] + a[10] * b[6] + a[14] * b[7];
      result[7]  = a[3] * b[4] + a[7] * b[5] + a[11] * b[6] + a[15] * b[7];

      result[8]  = a[0] * b[8] + a[4] * b[9] + a[8]  * b[10]+ a[12] * b[11];
      result[9]  = a[1] * b[8] + a[5] * b[9] + a[9]  * b[10]+ a[13] * b[11];
      result[10] = a[2] * b[8] + a[6] * b[9] + a[10] * b[10]+ a[14] * b[11];
      result[11] = a[3] * b[8] + a[7] * b[9] + a[11] * b[10]+ a[15] * b[11];

      result[12] = a[0] * b[12]+ a[4] * b[13]+ a[8]  * b[14]+ a[12] * b[15];
      result[13] = a[1] * b[12]+ a[5] * b[13]+ a[9]  * b[14]+ a[13] * b[15];
      result[14] = a[2] * b[12]+ a[6] * b[13]+ a[10] * b[14]+ a[14] * b[15];
      result[15] = a[3] * b[12]+ a[7] * b[13]+ a[11] * b[14]+ a[15] * b[15];

      return result;
    }
  };

  template<typename t>
  struct vec2_t {
  public:
    union {
      struct {
        t x {};
        t y {};
      };

      t buffer[2];
    };

    inline t &operator[](std::size_t index) {
      return this->buffer[index];
    }

    t *data() {
      return this->buffer;
    }
  public:
    inline vec2_t() = default;

    inline vec2_t(t _x, t _y) {
      this->x = _x;
      this->y = _y;
    }

    inline bicudo::vec2_t<t> operator/(t divisor) {
      return bicudo::vec2_t<t> {
        this->x / divisor,
        this->y / divisor
      };
    }

    inline bicudo::vec2_t<t> operator/(const bicudo::vec2_t<t> &r) {
      return bicudo::vec2_t<t> {
        this->x / r.x,
        this->y / r.y
      };
    }

    template<typename s>
    inline bicudo::vec2_t<t> operator+(const bicudo::vec2_t<s> &r) {
      return bicudo::vec2_t<t> {
        this->x + r.x,
        this->y + r.y
      };
    }

    template<typename s>
    inline bicudo::vec2_t<t> operator+(s val) {
      return bicudo::vec2_t<t> {
        this->x + val,
        this->y + val
      };
    }

    inline void operator+=(t val) {
      this->x += val;
      this->y += val;
    }

    template<typename s>
    inline void operator+=(const bicudo::vec2_t<s> &r) {
      this->x += r.x;
      this->y += r.y;
    }

    template<typename s>
    inline bicudo::vec2_t<t> operator-(const bicudo::vec2_t<s> &r) {
      return bicudo::vec2_t<t> {
        this->x - r.x,
        this->y - r.y
      };
    }

    template<typename s>
    inline bicudo::vec2_t<t> operator-(s val) {
      return bicudo::vec2_t<t> {
        this->x - val,
        this->y - val
      };
    }

    inline void operator-=(t val) {
      this->x -= val;
      this->y -= val;
    }

    template<typename s>
    inline void operator-=(const bicudo::vec2_t<s> &r) {
      this->x -= r.x;
      this->y -= r.y;
    }

    template<typename s>
    inline bicudo::vec2_t<t> operator*(s scalar) {
      return bicudo::vec2_t<t> {
        this->x * scalar,
        this->y * scalar
      };
    }

    template<typename s>
    inline bicudo::vec2_t<t> operator*(bicudo::vec2_t<s> scalar) {
      return bicudo::vec2_t<t> {
        this->x * scalar.x,
        this->y * scalar.y
      };
    }

    inline void operator*=(t scalar) {
      this->x *= scalar;
      this->y *= scalar;
    }

    inline t magnitude_no_sq() {
      return (this->x * this->x + this->y * this->y);
    }

    inline t magnitude() {
      return sqrtf(this->magnitude_no_sq());
    }

    template<typename s>
    inline t dot(const bicudo::vec2_t<s> &r) {
      return (this->x * r.x + this->y * r.y);
    }

    template<typename s>
    inline t cross(const bicudo::vec2_t<s> &r) {
      return (this->x * r.y - this->y * r.x);
    }

    template<typename s>
    inline t distance(const bicudo::vec2_t<s> &r) {
      return (*this - r).magnitude();
    }

    inline bicudo::vec2_t<t> normalize() {
      t len {this->magnitude()};
      if (len > 0.0f) {
        len = 1.0f / len;
      }

      return bicudo::vec2_t<t> {
        this->x * len,
        this->y * len
      };
    }

    template<typename s>
    inline bicudo::vec2_t<t> rotate(s a) {
      a = bicudo_deg2rad(a);
      return bicudo::vec2_t<t> {
        this->x * cosf(a) - this->y * sinf(a),
        this->x * sinf(a) + this->y * cosf(a)
      };
    }

    template<typename s>
    inline bicudo::vec2_t<t> rotate(s a, const bicudo::vec2_t<s> &center) {
      bicudo::vec2_t<t> displacement {
        *this - center
      };

      return displacement.rotate(a) + center;
    }

    inline bicudo::vec2_t<t> operator-() {
      return bicudo::vec2_t<t> {
        -this->x,
        -this->y
      };
    }

    template<typename s>
    inline bool operator==(const bicudo::vec2_t<s> &r) {
      return this->x == r.x && this->y && r.y; 
    }

    template<typename s>
    inline bool operator!=(const bicudo::vec2_t<s> &r) {
      return !(*this == r);
    }
  };

  template<typename t>
  struct vec3_t {
  public:
    union {
      struct {
        t x {};
        t y {};
        t z {};
      };

      t buffer[3];
    };

    inline t &operator[](std::size_t index) {
      return this->buffer[index];
    }

    t *data() {
      return this->buffer;
    }
  public:
    inline vec3_t() = default;

    inline vec3_t(t _x, t _y, t _z) {
      this->x = _x;
      this->y = _y;
      this->z = _z;
    }

    inline bicudo::vec3_t<t> normalize() {
      t len {std::sqrt(this->x * this->x + this->y * this->y + this->z * this->z)};
      if (len > 0) {
        len = 1.0f / len;
      }

      return bicudo::vec3_t<t> {
        this->x * len,
        this->y * len,
        this->z * len
      };
    }
  };

  template<typename t>
  struct vec4_t {
  public:
    union {
      struct {
        t x {};
        t y {};
        t z {};
        t w {};
      };

      t buffer[4];
    };

    inline t &operator[](std::size_t index) {
      return this->buffer[index];
    }

    t *data() {
      return this->buffer;
    }
  public:
    inline vec4_t() = default;

    inline vec4_t(t _x, t _y, t _z, t _w) {
      this->x = _x;
      this->y = _y;
      this->z = _z;
      this->w = _w;
    }

    template<typename s>
    inline bicudo::vec4_t<s> operator*(bicudo::mat4_t<s> r) {
      return bicudo::vec4_t<s> {
        r[0]  * this->x + r[1]  * this->y + r[2]  * this->z + r[3]  * this->w,
        r[4]  * this->x + r[5]  * this->y + r[6]  * this->z + r[7]  * this->w,
        r[8]  * this->x + r[9]  * this->y + r[10] * this->z + r[11] * this->w,
        r[12] * this->x + r[13] * this->y + r[14] * this->z + r[15] * this->w
      };
    }
  };

  template<typename t>
  struct edge_t {
  public:
    bicudo::vec2_t<t> a {};
    bicudo::vec2_t<t> b {};
  };

  template<typename t>
  void splash_vertices(
    bicudo::vec2_t<t> *p_vertices,
    bicudo::vec2_t<t> &pos,
    bicudo::vec2_t<t> &size
  ) {

  }
  
  template<typename t>
  void splash_edges_normalized(
    bicudo::vec2_t<t> *p_edges,
    bicudo::vec2_t<t> *p_vertices
  ) {

  }

  /**
   * Check: https://en.wikipedia.org/wiki/Orthographic_projection
   **/
  template<typename t>
  bicudo::mat4_t<t> ortho(
    t left,
    t right,
    t bottom,
    t top
  ) {
    t far {static_cast<t>(1)};
    t near {-(far)};

    t z0 {};
    t n2 {static_cast<t>(2)};

    t m11 {
      n2 / (right-left)
    };

    t m22 {
      n2 / (top-bottom)
    };

    t m33 {
      (-n2) / (far-near)
    };

    t m41 {
      -((right+left) / (right-left))
    };

    t m42 {
      -((top + bottom) / (top-bottom))
    };

    t m43 {
      -((far+near) / (far-near))
    };

    t m44 {
      static_cast<t>(1)
    };

    return bicudo::mat4_t<t> {
      m11, z0,  z0,  z0,
      z0,  m22, z0,  z0,
      z0,  z0,  m33, z0,
      m41, m42, m43, m44
    };
  }

  /**
   * Check: https://en.wikipedia.org/wiki/Rotation_matrix
   **/
  template<typename t>
  bicudo::mat4_t<t> rotate(
    bicudo::mat4_t<t> mat,
    bicudo::vec3_t<t> axis,
    t angle
  ) {
    angle = bicudo_deg2rad(angle);

    if (axis.z > 0.0f) {
      bicudo::mat4_t<t> rotate(
        cosf(angle),  sinf(angle), 0.0f, 0.0f,
        -sinf(angle), cosf(angle), 0.0f, 0.0f,
        0.0f,         0.0f,        1.0f, 0.0f,
        0.0f,         0.0f,        0.0f, 1.0f
      );

      return mat * rotate;
    }
  }

  template<typename t>
  bicudo::mat4_t<t> scale(
    bicudo::mat4_t<t> mat,
    bicudo::vec3_t<t> scale
  ) {
    mat._11 *= scale.x;
    mat._22 *= scale.y;
    mat._33 *= scale.z;
    return mat;
  }

  template<typename t>
  bicudo::mat4_t<t> translate(
    bicudo::mat4_t<t> mat,
    bicudo::vec2_t<t> pos
  ) {
    bicudo::mat4_t<float> translate(1.0f);
    translate._41 = pos.x;
    translate._42 = pos.y;
    translate._43 = 0.0f;
    return mat * translate;
  }

  template<typename t>
  t lerp(t a, t b, t dt) {
    return a + (b - a) * dt;
  };

  template<typename t>
  bicudo::vec2_t<t> lerp(
    const bicudo::vec2_t<t> &a,
    const bicudo::vec2_t<t> &b,
    t dt
  ) {
    return bicudo::vec2_t<t>(
      bicudo::lerp<t>(a.x, b.x, dt),
      bicudo::lerp<t>(a.y, b.y, dt)
    );
  }

  template<typename t>
  bicudo::vec2_t<t> lerp(
    const bicudo::vec2_t<t> &a,
    t b,
    t dt
  ) {
      a.x + (b - a.x) * dt,
    return bicudo::vec2_t<t>(
      bicudo::lerp<t>(a.x, b, dt),
      bicudo::lerp<t>(a.y, b, dt)
    );
  }

  template<typename t>
  bool aabb_collide_with_aabb(
    const bicudo::vec4_t<t> &a,
    const bicudo::vec4_t<t> &b
  ) {
    return true;
  }

  template<typename t>
  bool aabb_collide_with_vec2(
    const bicudo::vec2_t<t> &min,
    const bicudo::vec2_t<t> &max,
    const bicudo::vec2_t<t> &vec2
  ) {
    return (
      vec2.x > min.x && vec2.y > min.y && vec2.x < max.x && vec2.y < max.y
    );
  }

  template<typename t>
  bool vec4_collide_with_vec2(
    const bicudo::vec4_t<t> &vec4,
    const bicudo::vec2_t<t> &vec2
  ) {
    return (
      vec2.x > vec4.x && vec2.x < vec4.x + vec4.z
      &&
      vec2.y > vec4.y && vec2.y < vec4.y + vec4.w
    );
  }

  template<typename t>
  bool vec4_collide_with_vec4(
    const bicudo::vec4_t<t> &a,
    const bicudo::vec4_t<t> &b
  ) {
    return (
      (a.x < b.x + b.z && a.x + a.z > b.x)
      &&
      (a.y < b.y + b.w && a.y + a.w > b.y)
    );
  }
}

#endif
