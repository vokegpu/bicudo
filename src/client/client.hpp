#ifndef BICUDO_CLIENT_HPP
#define BICUDO_CLIENT_HPP

#include "client/tools/pickup.hpp"

namespace client {
  extern struct application {
  public:
    client::tools::pickup_info_t pickup_info {};
  } app;
}

#endif