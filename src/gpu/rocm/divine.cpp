#include <bicudo/gpu/rocm/divine.hpp>
#include <bicudo/core/core.hpp>

bicudo::gpu_rm_divine_kernel_t &bicudo::as_kernel(bicudo::gpu_rm_divine_pipeline_t &pipeline) {
 return pipeline.kernels.emplace_back();
}

bicudo::gpu_rm_divine_module_t &bicudo::as_module(
  bicudo::gpu_rm_divine_pipeline_t &pipeline,
  std::size_t index
) {
  if (index >= pipeline.kernels.size()) {
    bicudo::logw(
      bicudo::logtp(pipeline),
      "Could not get module by index, invalid index '", index, "' - out of range (", pipeline.kernels.size(), ") "
    );

    static bicudo::gpu_rm_divine_module_t not_found_kmodule {
      .unique_id = bicudo::found
    };

    return not_found_kmodule;
  }

  return pipeline.kernels.at(index);
}

bicudo::gpu_rm_divine_module_t &bicudo::as_module(
  bicudo::gpu_rm_divine_pipeline_t &pipeline,
  const std::string &tag
) {
  for (bicudo::gpu_rm_divine_module_t &kmodule : pipeline.kernels) {
    if (kmodule.tag == tag) {
      return kmodule;
    }
  }

  bicudo::logw(bicudo::logtp(pipeline), "Could not get module by name - not found.");

  static bicudo::gpu_rm_divine_module_t not_found_kmodule {
    .unique_id = bicudo::found
  };

  return not_found_kmodule;
}

bicudo::gpu_rm_divine_fun_t &bicudo::as_function(
  bicudo::gpu_rm_divine_module_t &kmodule,
  std::size_t index
) {
  if (index >= kmodule.functions.size()) {
    bicudo::logw(
      bicudo::logtk(kmodule),
      "Could not get function by index, invalid index '", index, "' - out of range (", kmodule.functions.size(), ") "
    );

    static bicudo::gpu_rm_divine_fun_t not_found_fun {
      .unique_id = bicudo::found
    };

    return not_found_fun;
  }

  return kmodule.functions.at(index);
}

bicudo::gpu_rm_divine_fun_t &bicudo::as_function(
  bicudo::gpu_rm_divine_module_t &kmodule,
  const std::string &name
) {
  for (bicudo::gpu_rm_divine_fun_t &f : kmodule.functions) {
    if (f.entry_point.name == name) {
      return f;
    }
  }

  bicudo::logw(bicudo::logtk(kmodule), "Could not get function by name - not found.");

  static bicudo::gpu_rm_divine_fun_t not_found_fun {
    .unique_id = bicudo::found
  };

  return not_found_fun;
}
