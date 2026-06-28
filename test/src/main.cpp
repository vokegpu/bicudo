#include <cstdint>

#include <bicudo/bicudo.hpp>
#include <bicudo/pipeline/rocm.hpp>

#include <bicudo/math/geometry.hpp>
#include <vector>

struct fun_memory_properties_t {
public:
  std::size_t shared_mem_bytes;
  std::size_t mem_size;
  hipStream_t stream;
};

struct fun_dimension_t {
public:
  bicudo::vec3_t<uint32_t> grid;
  bicudo::vec3_t<uint32_t> block;
};

struct fun_entry_point_t {
public:
  std::string name;
  hipFunction_t hip_function;
};

struct fun_args_t {
public:
  std::vector<void*> buffers;
};

struct fun_t {
public:
  fun_entry_point_t entry_point {};
  fun_dimension_t dimension {};
  fun_memory_properties_t memory {};
  fun_args_t args {};
};

struct kernel_t {
public:
  std::string tag {};
  std::string src {};
  hipModule_t hip_module {};
  hiprtcProgram hip_program {};
  std::vector<fun_t> functions {};
};

struct pipeline_t {
public:
  std::string tag {};
  std::vector<kernel_t> kernels {};
};

int32_t main(int32_t, char**) {
  bicudo::init_core_t bicudo_init_core = {
    .p_base = bicudo::rocm()
  };

  bicudo::core_t core {};
  bicudo::init(bicudo_init_core, core);

  pipeline_t pipeline {
    .tag = "meow amo vc lamed",
    .kernels = {
      kernel_t {
        .functions = {
          {
            .entry_point = { .name = "meow" }
          }
        }
      }
    }
  };

  auto j = 1;

  bicudo_hip_assert(
    j, hipSuccess, bicudo::loge("Failed!")
  );

  bicudo::flush();

  return 0;
}
