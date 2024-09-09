#include "immediate.hpp"
#include <iostream>

void bicudo::immediate_graphics::create() {
  bicudo::gpu_compile_shader_program(
    &program,
    {
      {
        GL_VERTEX_SHADER,
        R"(
        #version 450 core

        layout (location = 0) in vec2 aPos; 

        uniform vec4 uRect;
        uniform mat4 uProjection;

        out vec2 vUV;
        out vec4 vRect;

        void main() {
          gl_Position = uProjection * vec4((aPos * uRect.zw) + uRect.xy, 0.0f, 1.0f);
          vRect = uRect;
          vUV = aPos;
        }
        )"
      },
      {
        GL_FRAGMENT_SHADER,
        R"(
        #version 450 core

        layout (location = 0) out vec4 vFragColor;
        layout (binding = 0) uniform sampler2D uSampler;

        uniform bool uSamplerEnabled;
        uniform vec4 uColor;

        in vec2 vUV;
        in vec4 vRect;

        void main() {
          vec4 outColor = uColor;
          if (uSamplerEnabled) {
            outColor = texture(uSampler, vUV);
          }

          vFragColor = outColor;
        })"
      }
    }
  );

  this->uniform.linked_program = this->program;

  this->draw_call.polygon_type = GL_TRIANGLES;
  this->draw_call.index_type = GL_UNSIGNED_BYTE;
  this->draw_call.mode = bicudo::types::INDEXED;
  this->draw_call.offset = 0;
  this->draw_call.size = 6;
  this->draw_call.buffers.resize(2);

  glCreateVertexArrays(1, &this->draw_call.vao);
  glCreateBuffers(2, this->draw_call.buffers.data());

  glBindVertexArray(this->draw_call.vao);

  float vertices[8] {
    0.0f, 0.0f,
    1.0f, 0.0f,
    0.0f, 1.0f,
    1.0f, 1.0f
  };

  glBindBuffer(GL_ARRAY_BUFFER, this->draw_call.buffers.at(0));
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, (void*) 0);

  uint8_t indices[6] {
    0, 1, 3,
    3, 2, 0
  };

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->draw_call.buffers.at(1));
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  glBindVertexArray(0);
}

void bicudo::immediate_graphics::set_viewport(int32_t w, int32_t h) {
  this->mat4x4_projection = bicudo::ortho(
    0.0f,
    static_cast<float>(w),
    static_cast<float>(h),
    0.0f
  );

  glProgramUniformMatrix4fv(
    this->program,
    this->uniform["uProjection"],
    1,
    GL_FALSE,
    this->mat4x4_projection.data()
  );
}

void bicudo::immediate_graphics::invoke() {
  glUseProgram(this->program);
  glBindVertexArray(this->draw_call.vao);

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void bicudo::immediate_graphics::draw(
  bicudo::vec4 rect,
  bicudo::vec4 color,
  uint32_t bind_texture 
) {
  glUniform4fv(
    this->uniform["uRect"],
    1,
    rect.data()
  );

  glUniform4fv(
    this->uniform["uColor"],
    1,
    color.data()
  );

  if (bind_texture > 0) {
    glUniform1i(
      this->uniform["uSamplerEnabled"],
      true
    );

    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, bind_texture);
  }

  bicudo::gpu_dispatch_draw_call(&this->draw_call);
}

void bicudo::immediate_graphics::revoke() {
  glUseProgram(0);
  glBindVertexArray(0);
}