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

#define GLM_ENABLE_EXPERIMENTAL

#include <base/Ray.hpp>
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/normal.hpp>
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
  uint32_t iY = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t sX = blockDim.x * gridDim.x;
  uint32_t sY = blockDim.y * gridDim.y;

  for (uint32_t y = iY; y < _h; y += sY) {
    for (uint32_t x = iX; x < _w; x += sX) {
      float lPixX      = (2 * ((x + 0.5) / _w) - 1) * _scale * _ratio;
      float lPixY      = (1 - 2 * ((y + 0.5) / _h)) * _scale;
      vec3  lDirection = _mat * vec4(lPixX, lPixY, -1, 0.0f);

      _rays[y * _w + x].set(_pos, normalize(lDirection), x, y);
    }
  }
}

extern "C" __global__ void kTraceRay(Ray *    _rays,
                                     uint8_t *_img,
                                     BVHNode  _nodes,
                                     uint32_t _rootNode,
                                     MeshRaw  _mesh,
                                     vec3     _light,
                                     uint32_t _w,
                                     uint32_t _h) {
  uint32_t iX = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t iY = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t sX = blockDim.x * gridDim.x;
  uint32_t sY = blockDim.y * gridDim.y;

  for (uint32_t y = iY; y < _h; y += sY) {
    for (uint32_t x = iX; x < _w; x += sX) {
      Ray                lRay      = _rays[y * _w + x];
      CUDAPixel          lRes      = {121, 167, 229, 0};
      static const float lInfinity = HUGE_VALF;

      /*
       * Algorithm from:
       *
       * Attila T. Áfra and László Szirmay-Kalos. “Stackless Multi-BVH Traversal for CPU,
       * MIC and GPU Ray Tracing”. In: Computer Graphics Forum 33.1 (2014), pp. 129–140.
       * doi: 10.1111/cgf.12259. eprint: https://onlinelibrary.wiley.com/doi/pdf/
       * 10.1111/cgf.12259. url: https://onlinelibrary.wiley.com/doi/abs/10.1111/
       * cgf.12259.
       */

      uint64_t lBitStack_lo = 0;
      uint64_t lBitStack_hi = 0;
      uint32_t lNode        = _rootNode;

      Triangle lClosest = {0, 0, 0};
      float    lNearest = lInfinity;
      dvec2    lBarycentricTemp;

      float  lMinLeft;
      float  lMinRight;
      float  lTemp;
      double lDistance;

      while (true) {
        if (!_nodes.isLeaf(lNode)) {
          lRes.intCount++;
          uint32_t lLeft     = _nodes.left[lNode];
          uint32_t lRight    = _nodes.right[lNode];
          bool     lLeftHit  = _nodes.bbox[lLeft].intersect(lRay, 0.01f, lNearest + 0.01f, lMinLeft, lTemp);
          bool     lRightHit = _nodes.bbox[lRight].intersect(lRay, 0.01f, lNearest + 0.01f, lMinRight, lTemp);

          if (lLeftHit || lRightHit) {
            lBitStack_hi = (lBitStack_hi << 1) | (lBitStack_lo >> 63);
            lBitStack_lo <<= 1;

            if (lLeftHit && lRightHit) {
              lBitStack_lo |= 1;
              lNode = lMinLeft < lMinRight ? lLeft : lRight;
            } else {
              lNode = lLeftHit ? lLeft : lRight;
            }

            continue;
          }
        } else {
          for (uint32_t i = 0; i < _nodes.numFaces(lNode); ++i) {
            Triangle lTri = _mesh.faces[_nodes.beginFaces(lNode) + i];

            bool lHit = intersectRayTriangle<double>(static_cast<dvec3 const &>(lRay.getOrigin()),
                                                     static_cast<dvec3 const &>(lRay.getDirection()),
                                                     static_cast<dvec3 const &>(_mesh.vert[lTri.v1]),
                                                     static_cast<dvec3 const &>(_mesh.vert[lTri.v2]),
                                                     static_cast<dvec3 const &>(_mesh.vert[lTri.v3]),
                                                     lBarycentricTemp,
                                                     lDistance);

            if (lHit && lDistance < lNearest) {
              lNearest = lDistance;
              lClosest = lTri;
            }
          }
        }

        // Backtrac
        while ((lBitStack_lo & 1) == 0) {
          if (lBitStack_lo == 0 && lBitStack_hi == 0) { goto LABEL_END; } // I know, I know...
          lNode        = _nodes.parent[lNode];
          lBitStack_lo = (lBitStack_lo >> 1) | (lBitStack_hi << 63);
          lBitStack_hi >>= 1;
        }

        lNode = _nodes.isRightChild(lNode) ? _nodes.left[_nodes.parent[lNode]] : _nodes.right[_nodes.parent[lNode]];
        lBitStack_lo ^= 1;
      }

    LABEL_END:

      if (lNearest < lInfinity) {
        vec3  lNorm     = triangleNormal(_mesh.vert[lClosest.v1], _mesh.vert[lClosest.v2], _mesh.vert[lClosest.v3]);
        vec3  lHitPos   = lRay.getOrigin() + lNearest * lRay.getDirection();
        vec3  lLightDir = normalize(_light - lHitPos);
        float lDiffuse  = 1.0f + dot(lNorm, lLightDir);
        lDiffuse        = lDiffuse > 0.0f ? lDiffuse : 0.0f;

        lRes.r = lRes.g = lRes.b = static_cast<uint8_t>(lDiffuse * 127.0f);
      }

      reinterpret_cast<CUDAPixel *>(_img)[y * _w + x] = lRes;
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

extern "C" void tracerImage(Ray *    _rays,
                            uint8_t *_img,
                            BVHNode  _nodes,
                            uint32_t _rootNode,
                            MeshRaw  _mesh,
                            vec3     _light,
                            uint32_t _w,
                            uint32_t _h) {
  if (!_rays || !_img) { return; }
  dim3 lBlock(16, 16, 1);
  dim3 lGrid((_w + lBlock.x - 1) / lBlock.x, (_h + lBlock.y - 1) / lBlock.y);

  kTraceRay<<<lGrid, lBlock>>>(_rays, _img, _nodes, _rootNode, _mesh, _light, _w, _h);
}


extern "C" void copyImageToHost(CUDAPixel *_hostPixel, uint8_t *_cudaImg, uint32_t _w, uint32_t _h) {
  cudaMemcpy(_hostPixel, _cudaImg, _w * _h * sizeof(CUDAPixel), cudaMemcpyDeviceToHost);
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
  CUDA_RUN(cudaMalloc(_img, _w * _h * sizeof(CUDAPixel)));
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