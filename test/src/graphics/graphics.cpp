#include "graphics.hpp"
#include "meow.hpp"
#include <iostream>

void meow::immediate_graphics::create() {
  meow::gpu_compile_shader_program(
    &program,
    {
      {
        GL_VERTEX_SHADER,
        R"(
        #version 450 core

        layout (location = 0) in vec2 aPos; 

        uniform mat4 uRotate;
        uniform vec4 uRect;
        uniform mat4 uProjection;

        out vec2 vUV;
        out vec4 vRect;

        void main() {
          gl_Position = uProjection * (uRotate * vec4((aPos * uRect.zw) + uRect.xy, 0.0f, 1.0f));
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

        uniform int uSamplerEnabled;
        uniform vec4 uColor;
        uniform float uInf;
        uniform vec2 uSsize;

        in vec2 vUV;
        in vec4 vRect;

float rand(vec2 co) {
    return fract(sin(dot(co, vec2(12.9898, 78.233))) * 43758.5453);
}

vec3 hash3(vec3 n) {
    return fract(sin(n) * 1399763.5453123);
}

vec3 hpos(vec3 n) {
    return hash3(vec3(dot(n, vec3(157.0, 113.0, 271.0)), dot(n, vec3(311.0, 337.0, 179.0)), dot(n, vec3(271.0, 557.0, 431.0))));
}

vec4 mod289(vec4 x) {
    return x - floor(x * (1.0 / 289.0)) * 289.0;
}

vec4 permute(vec4 x) {
    return mod(((x * 34.0) + 1.0) * x, 289.0);
}

vec4 taylorInvSqrt(vec4 r) {
    return 1.79284291400159 - 0.85373472095314 * r;
}

vec2 fade(vec2 t) {
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

float noise4D(vec2 p) {
    vec4 Pi = floor(p.xyxy) + vec4(0.0, 0.0, 0.0, 1.0);
    vec4 Pf = fract(p.xyxy) - vec4(0.0, 0.0, 1.0, 1.0);

    Pi = mod289(Pi);

    vec4 ix = Pi.xzxz;
    vec4 iy = Pi.yyww;
    vec4 fx = Pf.xzxz;
    vec4 fy = Pf.yyww;

    vec4 i = permute(permute(ix) + iy);
    vec4 gx = fract(i * (1.0 / 41.0)) * 2.0 - 1.0;
    vec4 gy = abs(gx) - 0.5;
    vec4 tx = floor(gx + 0.5);
    gx = gx - tx;

    vec2 g00 = vec2(gx.x, gy.x);
    vec2 g10 = vec2(gx.y, gy.y);
    vec2 g01 = vec2(gx.z, gy.z);
    vec2 g11 = vec2(gx.w, gy.w);
    vec4 norm = taylorInvSqrt(vec4(dot(g00, g00), dot(g01, g01), dot(g10, g10), dot(g11, g11)));
    g00 *= norm.x;
    g01 *= norm.y;
    g10 *= norm.z;
    g11 *= norm.z;

    float n00 = dot(g00, vec2(fx.x, fy.y));
    float n10 = dot(g10, vec2(fx.y, fy.y));
    float n01 = dot(g01, vec2(fx.z, fy.z));
    float n11 = dot(g11, vec2(fx.w, fy.w));

    vec2 fade_xy = fade(Pf.xy);
    vec2 n_x = mix(vec2(n00, n01), vec2(n10, n11), fade_xy.x);
    float n_xy = mix(n_x.x, n_x.y, fade_xy.y);
    return 2.3 * n_xy;
}

float noise(vec2 st) {
    vec2 i = floor(st);
    vec2 f = fract(st);

    float a = rand(i);
    float b = rand(i + vec2(1.0, 0.0));
    float c = rand(i + vec2(0.0, 1.0));
    float d = rand(i + vec2(1.0, 1.0));

    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(a, b, u.x) + (c - a) * u.y * (1.0 - u.x) + (d - b) * u.x * u.y;
}

void main() {
    float j = uInf * 0.002f;

    if (uSamplerEnabled == 1) {
      float f  = (1.0 / tan(uInf)) * noise(vec2(length((rand(vec2(-2.0f, 663.0f))) * (((gl_FragCoord.xy) - vec2(uSsize.x/2, uSsize.y/2))))));
      vFragColor = (vec4(fract(f), noise(vec2((vRect) - length(rand(vec2(663.0f, f))))), f, 1.0f));
    } else if (uSamplerEnabled == 0){
      float f  = 1.0f - noise(vec2(length((rand(vec2(-2.0f, 663.0f))) * (((gl_FragCoord.xy) - vec2(uSsize.x/2, uSsize.y/2) * tan(uInf) * sin(uInf) * fract(uInf)) * tan(uInf)) * 4.0f - vRect.xy * uColor.x)));
      vFragColor = (vec4(fract(1.0f - f), noise(vec2((vec4(500, 500, 0, 1.0) + vRect) - fract(uInf) * length(rand(vec2(-uInf, f))))), j, 1.0f));
      vFragColor = vFragColor / fract(uInf * 0.0004f);
    } else if (uSamplerEnabled == 3) {
      float f  = 0.5f - noise(vec2(length((rand(vec2(-2.0f, 30.0f))) * (((gl_FragCoord.xy) - vec2(uSsize.x/2, uSsize.y/2))) * 10.0f)));
      vFragColor =vec4(1.0f, 1.0f, 1.0f, f);
    } else if (uSamplerEnabled == 2) {
      float f  = 1.0f - noise(vec2(length((rand(vec2(-2.0f, 30.0f))) * ((sin(gl_FragCoord.xy) - vec2(uSsize.x/2, uSsize.y/2))) * 10.0f)));
      vFragColor =vec4(0.0f, 0.0f, 0.0f, f + 0.5f);
    }
}
)"
      }
    }
  );

  this->uniform.linked_program = this->program;
  this->uniform.registry("uSamplerEnabled");
  this->uniform.registry("uColor");
  this->uniform.registry("uRect");
  this->uniform.registry("uRotate");
  this->uniform.registry("uProjection");
  this->uniform.registry("uInf");
  this->uniform.registry("uSsize");

  this->draw_call.polygon_type = GL_TRIANGLES;
  this->draw_call.index_type = GL_UNSIGNED_BYTE;
  this->draw_call.mode = meow::gpu::mode::INDEXED;
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

void meow::immediate_graphics::set_viewport(int32_t w, int32_t h) {
  this->mat4x4_projection = bicudo::ortho<float>(
    0.0f,
    static_cast<float>(w),
    static_cast<float>(h),
    0.0f
  );

  this->viewport.x = 0.0f;
  this->viewport.y = 0.0f;
  this->viewport.z = static_cast<float>(w);
  this->viewport.w = static_cast<float>(h);

  meow::camera &camera {meow::app.camera};
  bicudo::vec2_t<float> center {this->viewport.z / 2, this->viewport.w / 2};

  bicudo::vec2_t<float> delta {(center / this->current_zoom) + camera.rect.pos};
  this->current_zoom = camera.zoom;
  camera.rect.pos = delta - (center / this->current_zoom);

  this->mat4x4_projection = bicudo::scale<float>(
    this->mat4x4_projection,
    {this->current_zoom, this->current_zoom, 1.0f}
  );  

  glProgramUniformMatrix4fv(
    this->program,
    this->uniform["uProjection"],
    1,
    GL_FALSE,
    this->mat4x4_projection.data()
  );
}

void meow::immediate_graphics::invoke() {
  glUseProgram(this->program);
  glBindVertexArray(this->draw_call.vao);

  glDisable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
}

void meow::immediate_graphics::draw(
  bicudo::vec4_t<float> rect,
  bicudo::vec4_t<float> color,
  float angle,
  uint32_t bind_texture
) {
  this->mat4x4_rotate = bicudo::mat4_t<float>(1.0f);

  if (!bicudo::assert_float(angle, 0.0f)) {
    bicudo::vec2_t<float> center {
      rect.x + (rect.z / 2), rect.y + (rect.w / 2)
    };

    this->mat4x4_rotate = bicudo::translate(this->mat4x4_rotate, center);
    this->mat4x4_rotate = bicudo::rotate(this->mat4x4_rotate, {0.0f, 0.0f, 1.0f}, angle);
    this->mat4x4_rotate = bicudo::translate(this->mat4x4_rotate, -center);
  }

  glUniformMatrix4fv(
    this->uniform["uRotate"],
    1,
    GL_FALSE,
    this->mat4x4_rotate.data()
  );

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

  glUniform1f(
    this->uniform["uInf"],
    this->uinf
  );

  glUniform2f(
    this->uniform["uSsize"],
    this->viewport.z,
    this->viewport.w
  );

  this->uinf += 0.0002f;

  if (bind_texture > 0) {
    glUniform1i(
      this->uniform["uSamplerEnabled"],
      bind_texture
    );

    //glActiveTexture(GL_TEXTURE0);
    //glBindTexture(GL_TEXTURE_2D, bind_texture);
  } else {
    glUniform1i(
      this->uniform["uSamplerEnabled"],
      0
    );
  }

  meow::gpu_dispatch_draw_call(&this->draw_call);
}

void meow::immediate_graphics::revoke() {
  glUseProgram(0);
  glBindVertexArray(0);
}
