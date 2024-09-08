#ifndef BICUDO_UTIL_MATH_HPP
#define BICUDO_UTIL_MATH_HPP

#include <math.h>
#include <cfloat>
#include <cstdint>

#define bicudo_min(a, b)            ((a) < (b) ? (b) : (a))
#define bicudo_max(a, b)            ((a) > (b) ? (b) : (a))
#define bicudo_clamp(val, min, max) ((val < min ? min : (val > max ? max : val)))

namespace bicudo {
  extern uint64_t framerate;
  extern uint64_t current_framerate;
  extern uint64_t cpu_interval_ms;
  extern float dt;

  void set_framerate(uint64_t wish_fps);

  constexpr bool assert_float(float x, float y) {
    return fabsf(x - y) <= FLT_EPSILON * fmaxf(fabsf(x), fabsf(y));
  }

  struct vec2 {
  public:
    union {
      struct {
        float x {};
        float y {};
      };

      float data[2];
    };
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

    inline bicudo::vec2 operator*(float scalar) {
      return bicudo::vec2 {
        this->x * scalar,
        this->y * scalar
      };
    }

    inline float magnitude_no_sq() {
      return (this->x * this->x + this->y * this->y);
    }

    inline float magnitude() {
      return std::sqrt(this->x * this->x + this->y * this->y);
    }

    inline float dot(const bicudo::vec2 &r) {
      return (this->x * r.y - this->y * r.x);
    }

    inline bicudo::vec2 normalize() {
      float len {this->magnitude()};
      if (len > 0) {
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
  };

  struct placement {
  public:
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
  };
}

#endif