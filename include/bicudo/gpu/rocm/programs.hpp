#ifndef BICUDO_GPU_ROCM_PROGRAMS_HPP
#define BICUDO_GPU_ROCM_PROGRAMS_HPP

namespace bicudo {
  const char* hip_rocm_kernel_runtime_assert {
    R"(
      /**
       * This hip runtime should perform memory-access assert
       * to ROCm runtime initialization.
       **/
      
      extern "C" __global__
      void gpu_divine_main(
        float *__restrict__ p_assert_buffer
      ) {
        p_assert_buffer[0] = 17.0f; // from 153.0f
        p_assert_buffer[1] = 27.0f; // from 26.0f
        p_assert_buffer[2] = 37.0f; // from 24.0f
        p_assert_buffer[3] = 47.0f; // from 6.0f
        p_assert_buffer[4] = 52.0f; // from 1977.0f

        int random = (threadIdx.x + threadIdx.y * blockDim.x);
        int body = blockIdx.x;

        // printf("%i, %i\n", body, random);
      }
    )"
  };

  const char* hip_rocm_kernel_collision_detection {
    R"(
      /**
       * This hip runtime should perform collision detection.
       **/

      #define MAX_WORLD_DISTANCE 99999.0f
      #define SHAPE_VERTICES_UNIT_COUNT 10

      struct vec2_t {
        float x {};
        float y {};
      };

      #define bicudo_vec2_set(region, index, unit, value) \
        p_hd_hypergroup_bodies[region + index * 2 + unit] = value;

      #define bicudo_vec2_at(region, index) \
        vec2_t { \
          .x = p_hd_hypergroup_bodies[region + (index * 2) + 0], \
          .y = p_hd_hypergroup_bodies[region + (index * 2) + 1] \
        }

      #define bicudo_vec2_normalize(v) \
        len = hypotf(v.x, v.y); \
        if (len > 0.0f) { \
          len = 1.0f / len; \
        } \
        v.x *= len; \
        v.y *= len;

      #define bicudo_vec2_subtract(v1, v2) \
        vec2_t {.x = v1.x - v2.x, .y = v1.y - v2.y}

      #define bicudo_vec2_sum(v1, v2) \
        vec2_t {.x = v1.x + v2.x, .y = v1.y + v2.y}
      
      #define bicudo_vec2_scale(v1, s) \
        vec2_t {.x = v1.x * s, .y = v1.y * s}
      
      #define bicudo_vec2_dot(v1, v2) \
        (v1.x * v2.x + v1.y * v2.y)

      extern "C" __global__
      void gpu_divine_main(
        float *__restrict__ p_hd_hypergroup_bodies
      ) {
        int random = (threadIdx.x + threadIdx.y * blockDim.x);
        int body = blockIdx.x;

        random *= SHAPE_VERTICES_UNIT_COUNT;
        body *= SHAPE_VERTICES_UNIT_COUNT;

        //printf("body: %i, random: %i\n", body, random);
        if (body == random) return;

        bool has {true};
        float len {};

        /**
         * We check for both side (f == 0 & f == 1) :
         * (A & B) & (B & A) to validade a collision.
         **/

        int f {};
        for (f = 0; f < 2; f++) {
          int a = f == 0 ? body : random;
          int b = f == 0 ? random : body;

          vec2_t normals[4] {
            bicudo_vec2_subtract(bicudo_vec2_at(a, 1), bicudo_vec2_at(a, 2)),
            bicudo_vec2_subtract(bicudo_vec2_at(a, 2), bicudo_vec2_at(a, 3)),
            bicudo_vec2_subtract(bicudo_vec2_at(a, 3), bicudo_vec2_at(a, 0)),
            bicudo_vec2_subtract(bicudo_vec2_at(a, 0), bicudo_vec2_at(a, 1))
          };

          float bestdist {MAX_WORLD_DISTANCE};
          float maxdist {};

          int i {};
          int j {};
          int best {};

          vec2_t point {};
          vec2_t bestpoint {};

          for (i = 0; has && i < 4; i++) {
            vec2_t normal = normals[i];
            bicudo_vec2_normalize(normal);
            normal = bicudo_vec2_scale(normal, -1.0f);
            vec2_t v1 = bicudo_vec2_at(a, i);

            maxdist = -MAX_WORLD_DISTANCE;
            has = false;

            for (j = 0; j < 4; j++) {
              vec2_t v2 = bicudo_vec2_at(b, j);
              vec2_t dir = 
                bicudo_vec2_subtract(
                  v2,
                  v1
                );

              float proj = bicudo_vec2_dot(dir, normal);
              if (proj > 0 && proj > maxdist) {
                maxdist = proj;
                point = v2;
                has = true;
              }
            }

            if (has && maxdist < bestdist) {
              bestdist = maxdist;
              best = i;
              bestpoint = point;
            }
          }

          if (has) {
            continue;
          }

          return;
        }

        bicudo_vec2_set(body, 4, 0, 1.0f);
      }
    )"
  };
}

#endif

//
