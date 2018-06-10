/*
 * Copyright (C) 2018 Daniel Mensinger
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#define GLM_FORCE_NO_CTOR_INIT

#include <glm/vec3.hpp>

#ifdef __CUDACC__
#  ifndef CUDA_CALL
#    define CUDA_CALL __host__ __device__ __forceinline__
#  endif
#else
#  ifndef CUDA_CALL
#    define CUDA_CALL inline
#  endif
#endif

namespace BVHTest {
namespace base {

using glm::vec3;

class alignas(16) Ray final {
 public:
  struct alignas(4) Sign {
    uint8_t x;
    uint8_t y;
    uint8_t z;
  };

 private:
  vec3 vPos;
  vec3 vDir;
  vec3 vInvDir;
  Sign vSign;

  uint32_t vPX = UINT32_MAX;
  uint32_t vPY = UINT32_MAX;

 public:
  CUDA_CALL Ray() {}
  CUDA_CALL Ray(vec3 _pos, vec3 _dir) : Ray(_pos, _dir, UINT32_MAX, UINT32_MAX) {}
  CUDA_CALL Ray(vec3 _pos, vec3 _dir, uint32_t _x, uint32_t _y) : vPos(_pos), vDir(_dir), vPX(_x), vPY(_y) {
    vInvDir = 1.0f / vDir;
    vSign.x = vInvDir.x < 0 ? 1 : 0;
    vSign.y = vInvDir.y < 0 ? 1 : 0;
    vSign.z = vInvDir.z < 0 ? 1 : 0;
  }

  CUDA_CALL vec3 const &getOrigin() const { return vPos; }
  CUDA_CALL vec3 const &getDirection() const { return vDir; }
  CUDA_CALL vec3 const &getInverseDirection() const { return vInvDir; }
  CUDA_CALL Sign const &getSign() const { return vSign; }
};

} // namespace base
} // namespace BVHTest
