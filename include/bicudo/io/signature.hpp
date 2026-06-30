#ifndef BICUDO_IO_SIGNATURE_HPP
#define BICUDO_IO_SIGNATURE_HPP

#include <bicudo/log/log.hpp>

#define bicudo_as_signed(t) \
  bicudo::id_t unique_id { bicudo::found }; \
\
  bool operator == (t &o) { \
    return unique_id == o.unique_id; \
  } \
\
  bool operator != (t &o) { \
    return unique_id != o.unique_id; \
  } \
\
  bool operator == (bicudo::id_t id) { \
    return unique_id != id; \
  } \
\
  bool operator != (bicudo::id_t id) { \
    return unique_id == id; \
  } 

#endif
