#ifndef BICUDO_MATH_GEOMETRY_HPP
#define BICUDO_MATH_GEOMETRY_HPP

namespace bicudo {
  template<typename t>
  struct vec3_t {
    union {
      struct {
        t x {};
        t y {};
        t z {};
      };
    };
  public:
    inline vec3_t() = default;

    inline vec3_t(t _x, t _y, t _z) {
      this->x = _x;
      this->y = _y;
      this->z = _z;
    }

    // TODO: add non-useless vector 3 properties operators

    template<typename s>
    operator bicudo::vec3_t<s>() {
      return bicudo::vec3_t<s>{
        static_cast<s>(this->x),
        static_cast<s>(this->y),
        static_cast<s>(this->z)
      };
    }
  };
}

#endif
