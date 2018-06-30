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

#include <base/Ray.hpp>
#include <glm/mat4x4.hpp>
#include "CUDAKernels.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <iostream>

#define CUDA_RUN(call)                                                                                                 \
  lRes = call;                                                                                                         \
  if (lRes != cudaSuccess) {                                                                                           \
    cout << "CUDA ERROR (" << __FILE__ << ":" << __LINE__ << "): " << cudaGetErrorString(lRes) << endl;                \
    goto error;                                                                                                        \
  }

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::base;


extern "C" __global__ void kGenerateRays(
    Ray *_rays, uint32_t _w, uint32_t _h, mat4 _mat, vec3 _pos, float _ratio, float _scale) {
  uint32_t iX = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t iY = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t sX = blockDim.x * gridDim.x;
  uint32_t sY = blockDim.x * gridDim.x;

  for (uint32_t y = iY; y < _h; y += sY) {
    for (uint32_t x = iX; x < _w; x += sX) {
      float lPixX      = (2 * ((x + 0.5) / _w) - 1) * _scale * _ratio;
      float lPixY      = (1 - 2 * ((y + 0.5) / _h)) * _scale;
      vec3  lDirection = _mat * vec4(lPixX, lPixY, -1, 0.0f);

      _rays[y * _w + x].set(_pos, normalize(lDirection), x, y);
    }
  }
}

extern "C" void generateRays(Ray *_rays, uint32_t _w, uint32_t _h, vec3 _pos, vec3 _lookAt, vec3 _up, float _fov) {
  if (!_rays) { return; }

  mat4  lCamToWorld  = inverse(lookAtRH(_pos, _lookAt, _up));
  float lAspectRatio = static_cast<float>(_w) / static_cast<float>(_h);
  float lScale       = tan(radians(0.5 * _fov));

  dim3 lBlock(16, 16, 1);
  dim3 lGrid((_w + lBlock.x - 1) / lBlock.x, (_h + lBlock.y - 1) / lBlock.y);

  kGenerateRays<<<lGrid, lBlock>>>(_rays, _w, _h, lCamToWorld, _pos, lAspectRatio, lScale);
}



extern "C" bool allocateRays(Ray **_rays, uint32_t _num) {
  cudaError_t lRes;
  CUDA_RUN(cudaMalloc(_rays, _num * sizeof(Ray)));
  return true;

error:
  return false;
}

extern "C" bool allocateImage(uint8_t **_img, uint32_t _w, uint32_t _h) {
  cudaError_t lRes;
  CUDA_RUN(cudaMalloc(_img, _w * _h * 4 * sizeof(uint8_t)));
  return true;

error:
  return false;
}

extern "C" void freeRays(Ray **_rays) {
  if (*_rays) {
    cudaFree(*_rays);
    *_rays = NULL;
  }
}

extern "C" void freeImage(uint8_t *_img) {
  if (_img) { cudaFree(_img); }
}

extern "C" void tracerDoCudaSync() { cudaDeviceSynchronize(); }
extern "C" void initCUDA_GL() { cudaGLSetGLDevice(0); }
