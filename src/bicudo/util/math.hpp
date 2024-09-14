#ifndef BICUDO_UTIL_MATH_HPP
#define BICUDO_UTIL_MATH_HPP

#include <math.h>
#include <cfloat>
#include <vector>
#include <cstdint>

#define bicudo_clamp_min(a, b)      ((a) < (b) ? (b) : (a))
#define bicudo_clamp_max(a, b)      ((a) > (b) ? (b) : (a))
#define bicudo_clamp(val, min, max) (val < min ? min : ((max < val) ? max : val))
#define bicudo_deg2rad(x)           ((x) * 0.0174533f)

namespace bicudo {
  extern uint64_t framerate;
  extern uint64_t current_framerate;
  extern uint64_t cpu_interval_ms;
  extern float dt;

  void set_framerate(uint64_t wish_fps);

  constexpr bool assert_float(float x, float y) {
    return fabsf(x - y) <= FLT_EPSILON * fmaxf(fabsf(x), fabsf(y));
  }

  typedef struct vec2 {
  public:
    union {
      struct {
        float x {};
        float y {};
      };

      float buffer[2];
    };

    inline float &operator[](std::size_t index) {
      return this->buffer[index];
    }

    float *data() {
      return this->buffer;
    }
  public:
    inline vec2() = default;

    inline vec2(float _x, float _y) {
      this->x = _x;
      this->y = _y;
    }

    inline bicudo::vec2 operator+(const bicudo::vec2 &r) {
      return bicudo::vec2 {
        this->x + r.x,
        this->y + r.y
      };
    }

    inline bicudo::vec2 operator+(float val) {
      return bicudo::vec2 {
        this->x + val,
        this->y + val
      };
    }

    inline void operator+=(float val) {
      this->x += val;
      this->y += val;
    }

    inline void operator+=(const bicudo::vec2 &r) {
      this->x += r.x;
      this->y += r.y;
    }

    inline bicudo::vec2 operator-(const bicudo::vec2 &r) {
      return bicudo::vec2 {
        this->x - r.x,
        this->y - r.y
      };
    }

    inline bicudo::vec2 operator-(float val) {
      return bicudo::vec2 {
        this->x - val,
        this->y - val
      };
    }

    inline void operator-=(float val) {
      this->x -= val;
      this->y -= val;
    }

    inline void operator-=(const bicudo::vec2 &r) {
      this->x -= r.x;
      this->y -= r.y;
    }

    inline bicudo::vec2 operator*(float scalar) {
      return bicudo::vec2 {
        this->x * scalar,
        this->y * scalar
      };
    }

    inline void operator*=(float scalar) {
      this->x *= scalar;
      this->y *= scalar;
    }

    inline float magnitude_no_sq() {
      return (this->x * this->x + this->y * this->y);
    }

    inline float magnitude() {
      return std::sqrt(this->magnitude_no_sq());
    }

    inline float dot(const bicudo::vec2 &r) {
      return (this->x * r.x + this->y * r.y);
    }

    inline float cross(const bicudo::vec2 &r) {
      return (this->x * r.y - this->y * r.x);
    }

    inline float distance(const bicudo::vec2 &r) {
      return (*this - r).magnitude();
    }

    inline bicudo::vec2 normalize() {
      float len {this->magnitude()};
      if (len > 0.0f) {
        len = 1.0f / len;
      }

      return bicudo::vec2 {
        this->x * len,
        this->y * len
      };
    }

    inline bicudo::vec2 rotate(float a) {
      return bicudo::vec2 {
        this->x * cosf(a) - this->y * sinf(a),
        this->x * sinf(a) + this->y * cosf(a)
      };
    }

    inline bicudo::vec2 rotate(float a, const bicudo::vec2 &center) {
      bicudo::vec2 displacement {
        *this - center
      };

      return displacement.rotate(a) + center;
    }

    inline bicudo::vec2 operator-() {
      return bicudo::vec2 {
        -this->x,
        -this->y
      };
    }

    inline bool operator==(const bicudo::vec2 &r) {
      return bicudo::assert_float(this->x, r.x) && bicudo::assert_float(this->y, r.y); 
    }

    inline bool operator!=(const bicudo::vec2 &r) {
      return !(*this == r);
    }
  } vec2;

  typedef struct vec3 {
  public:
    union {
      struct {
        float x {};
        float y {};
        float z {};
      };

      float buffer[3];
    };

    inline float &operator[](std::size_t index) {
      return this->buffer[index];
    }

    float *data() {
      return this->buffer;
    }
  public:
    inline vec3() = default;

    inline vec3(float _x, float _y, float _z) {
      this->x = _x;
      this->y = _y;
      this->z = _z;
    }

    inline bicudo::vec3 normalize() {
      float len {std::sqrt(this->x * this->x + this->y * this->y + this->z * this->z)};
      if (len > 0) {
        len = 1.0f / len;
      }

      return bicudo::vec3 {
        this->x * len,
        this->y * len,
        this->z * len
      };
    }
  } vec3;

  typedef struct vec4 {
  public:
    union {
      struct {
        float x {};
        float y {};
        float z {};
        float w {};
      };

      float buffer[4];
    };

    inline float &operator[](std::size_t index) {
      return this->buffer[index];
    }

    float *data() {
      return this->buffer;
    }
  public:
    inline vec4() = default;

    inline vec4(float _x, float _y, float _z, float _w) {
      this->x = _x;
      this->y = _y;
      this->z = _z;
      this->w = _w;
    }
  } vec4;

  typedef struct mat4 {
  public:
    union {
      struct {
        float _11 {}, _12 {}, _13 {}, _14 {};
        float _21 {}, _22 {}, _23 {}, _24 {};
        float _31 {}, _32 {}, _33 {}, _34 {};
        float _41 {}, _42 {}, _43 {}, _44 {};
      };

      float buffer[16];
    };

    inline float &operator[](std::size_t index) {
      return this->buffer[index];
    }

    float *data() {
      return this->buffer;
    }
  public:
    inline mat4(float identity = 1.0f) {
      this->_11 = this->_22 = this->_33 = this->_44 = identity;
    }

    inline mat4(
      float f11, float f12, float f13, float f14,
      float f21, float f22, float f23, float f24,
      float f31, float f32, float f33, float f34,
      float f41, float f42, float f43, float f44
    ) {
      this->_11 = f11; this->_12 = f12; this->_13 = f13; this->_14 = f14;
      this->_21 = f21; this->_22 = f22; this->_23 = f23; this->_24 = f24;
      this->_31 = f31; this->_32 = f32; this->_33 = f33; this->_34 = f34;
      this->_41 = f41; this->_42 = f42; this->_43 = f43; this->_44 = f44;
    }

    inline bicudo::mat4 operator*(bicudo::mat4 &r) {
      bicudo::mat4 &a {*this};
      bicudo::mat4 &b {r};
      bicudo::mat4 result {};

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
  } mat4;

  struct placement {
  public:
    const char *p_tag {};

    float mass {};
    float friction {};
    float restitution {};
    float inertia {};

    bicudo::vec2 min {};
    bicudo::vec2 max {};

    bicudo::vec2 pos {};
    bicudo::vec2 size {};
    bicudo::vec2 velocity {};
    bicudo::vec2 acc {};

    float angle {};
    float angular_velocity {};
    float angular_acc {};

    std::vector<bicudo::vec2> vertices {};
    std::vector<bicudo::vec2> edges {};

    bool was_collided {};
  };

  void splash_vertices(bicudo::vec2 *p_vertices, bicudo::vec2 &pos, bicudo::vec2 &size);
  void splash_edges_normalized(bicudo::vec2 *p_edges, bicudo::vec2 *p_vertices);

  bicudo::mat4 ortho(float left, float right, float bottom, float top);
  bicudo::mat4 rotate(bicudo::mat4 mat, bicudo::vec3 axis, float angle);
  bicudo::mat4 translate(bicudo::mat4 mat, bicudo::vec2 pos);

  bool vec4_collide_with_vec2(const bicudo::vec4 &vec4, const bicudo::vec2 &vec2);
  void move(bicudo::placement *p_placement, const bicudo::vec2 &dir);
  void rotate(bicudo::placement *p_placement, float angle_dir);
  void mass(bicudo::placement *p_placement, float mass);
}

#endif