#ifndef BICUDO_GPU_CONTENT_HPP
#define BICUDO_GPU_CONTENT_HPP

namespace bicudo::gpu {
  struct buffer_t {
  public:
    uint64_t size {};
    void *p_device {};
    void *p_host {};
  };
}

#endif