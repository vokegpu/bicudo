#include "renderer.hpp"
#include "bicudo/util/log.hpp"

uint32_t &bicudo::gpu::uniform::operator[](std::string_view key) {
  uint32_t &location {this->program_location_map[key]};
  return (
    location = glGetUniformLocation(this->linked_program, key.data())
  );
}

bicudo::result bicudo::gpu_compile_shader_program(
  uint32_t *p_program,
  const std::vector<bicudo::gpu::shader> &shader_list
) {
  if (shader_list.empty()) {
    bicudo::log() << "Error: Invalid shader, empty resources";
    return bicudo::types::FAILED;
  }

  std::string shader_src {};
  std::vector<uint32_t> compiled_shader_list {};
  int32_t status {};

  uint32_t &program {*p_program};
  program = glCreateProgram();

  std::string msg {};
  for (const bicudo::gpu::shader &module : shader_list) {
    uint32_t shader {glCreateShader(module.stage)};
    const char *p_src {module.p_path_or_source};
    glShaderSource(shader, 1, &p_src, nullptr);
    glCompileShader(shader);

    glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
    if (status == GL_FALSE) {
      glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &status);
      msg.resize(status);
      glGetShaderInfoLog(shader, status, nullptr, msg.data());
      bicudo::log() << "Error: Failed to compile shader '" << p_src  << "' stage: \n" << msg;
      break;
    }

    compiled_shader_list.push_back(shader);
  }

  bool keep {compiled_shader_list.size() == shader_list.size()};

  for (uint32_t &shaders : compiled_shader_list) {
    if (keep) {
      glAttachShader(program, shaders);
    }

    glDeleteShader(shaders);
  }

  if (keep) {
    glLinkProgram(program);
    glGetProgramiv(program, GL_LINK_STATUS, &status);

    if (status == GL_FALSE) {
      glGetProgramiv(program, GL_INFO_LOG_LENGTH, &status);
      msg.resize(status);
      glGetProgramInfoLog(program, status, nullptr, msg.data());
      bicudo::log() << "Error: Failed to link program: \n" << msg;
    }
  }

  return bicudo::types::SUCCESS;
}

bicudo::result bicudo::gpu_dispatch_draw_call(
  bicudo::gpu::draw_call *p_draw_call
) {
  if (p_draw_call == nullptr) {
    return bicudo::types::FAILED;
  }

  switch (p_draw_call->mode) {
  case bicudo::types::INDEXED:
    glDrawElements(
      p_draw_call->polygon_type,
      p_draw_call->size,
      p_draw_call->index_type,
      (void*) p_draw_call->offset
    );
    break;
  case bicudo::types::ARRAYS:
    glDrawArrays(
      p_draw_call->polygon_type,
      p_draw_call->offset,
      p_draw_call->size
    );
    break;
  }

  return bicudo::types::SUCCESS;
}