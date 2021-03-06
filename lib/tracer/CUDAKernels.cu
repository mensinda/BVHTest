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

#include "BVHTestCfg.hpp"
#include <base/Ray.hpp>
#include <glm/gtx/normal.hpp>
#include <glm/mat4x4.hpp>
#include "CUDAKernels.hpp"
#include <glm/gtc/matrix_transform.hpp>
#include <cub/cub.cuh>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
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

struct DataShared {
  Triangle closest;
};

// source: https://github.com/hpicgs/cgsee/wiki/Ray-Box-Intersection-on-the-GPU
extern "C" __device__ __forceinline__ bool intersectRayAABB(AABB &_box, Ray const &_r, float t0, float t1, float &tmin) {
  glm::vec3 const &lOrigin = _r.getOrigin();
  glm::vec3 const &lInvDir = _r.getInverseDirection();
  Ray::Sign const &lSign   = _r.getSign();

  float tymin, tymax, tzmin, tzmax, tmax;

  tmin  = (_box.minMax[lSign.x].x - lOrigin.x) * lInvDir.x;
  tmax  = (_box.minMax[1 - lSign.x].x - lOrigin.x) * lInvDir.x;
  tymin = (_box.minMax[lSign.y].y - lOrigin.y) * lInvDir.y;
  tymax = (_box.minMax[1 - lSign.y].y - lOrigin.y) * lInvDir.y;
  tzmin = (_box.minMax[lSign.z].z - lOrigin.z) * lInvDir.z;
  tzmax = (_box.minMax[1 - lSign.z].z - lOrigin.z) * lInvDir.z;
  if (tymin > tmin) { tmin = tymin; }
  if (tzmin > tmin) { tmin = tzmin; }
  if (tymax < tmax) { tmax = tymax; }
  if (tzmax < tmax) { tmax = tzmax; }

  return ((tmin < tmax) && (tmin < t1) && (tmax > t0));
}

// Stolen from GLM -- then optimized for CUDA
extern "C" __device__ __forceinline__ bool intersectRayTriangle2(
    vec3 const &orig, vec3 const &dir, vec3 const &vert0, vec3 const &vert1, vec3 const &vert2, float &distance) {
  vec3 const edge1 = vert1 - vert0;
  vec3 const edge2 = vert2 - vert0;

  vec3 const  p   = glm::cross(dir, edge2);
  float const det = glm::dot(edge1, p);

  vec3 const tvec = orig - vert0;
  vec3 const qvec = glm::cross(tvec, edge1);

  float const inv_det = __fdividef(1.0f, det); // __fdividef is fast division

  float const u = glm::dot(tvec, p);
  float const v = glm::dot(dir, qvec);

  distance = glm::dot(edge2, qvec) * inv_det;

#if CUDA_FACE_CULLING
  return ((det > 0.0f) && ((u >= 0.0f && u <= det) && (v >= 0.0f && (u + v) <= det)));
#else
  return ((det > 0.0f) && ((u >= 0.0f && u <= det) && (v >= 0.0f && (u + v) <= det))) ||
         ((det < 0.0f) && ((u <= 0.0f && u >= det) && (v <= 0.0f && (u + v) >= det)));
#endif
}

extern "C" __global__ void kTraceRay(Ray *      _rays,
                                     CUDAPixel *_img,
                                     BVHNode *  _nodes,
                                     uint32_t   _rootNode,
                                     MeshRaw    _mesh,
                                     vec3       _light,
                                     uint32_t   _w,
                                     uint32_t   _h) {
  uint32_t iX = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t iY = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t sX = blockDim.x * gridDim.x;
  uint32_t sY = blockDim.y * gridDim.y;

  uint32_t lID = threadIdx.y * 8 + threadIdx.x;

  for (uint32_t y = iY; y < _h; y += sY) {
    for (uint32_t x = iX; x < _w; x += sX) {
      Ray       lRay = _rays[y * _w + x];
      CUDAPixel lRes = {0, 0, 0};

      __shared__ DataShared lEtcData[64];

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

      float lMinLeft;
      float lMinRight;
      float lDistance;
      float lNearest = HUGE_VALF;

      while (true) {
        if (!_nodes[lNode].isLeaf()) {
          lRes.intCount++;
          uint32_t lLeft     = _nodes[lNode].left;
          uint32_t lRigh     = _nodes[lNode].right;
          bool     lLeftHit  = intersectRayAABB(_nodes[lLeft].bbox, lRay, 0.01f, lNearest + 0.01f, lMinLeft);
          bool     lRightHit = intersectRayAABB(_nodes[lRigh].bbox, lRay, 0.01f, lNearest + 0.01f, lMinRight);

          if (lLeftHit || lRightHit) {
            lBitStack_hi = (lBitStack_hi << 1) | (lBitStack_lo >> 63);
            lBitStack_lo <<= 1;

            if (lLeftHit && lRightHit) {
              lBitStack_lo |= 1;
              lNode = lMinLeft < lMinRight ? lLeft : lRigh;
            } else {
              lNode = lLeftHit ? lLeft : lRigh;
            }

            continue;
          }
        } else {
          for (uint32_t i = 0; i < _nodes[lNode].numFaces(); ++i) {
            Triangle lTri = _mesh.faces[_nodes[lNode].beginFaces() + i];

            bool lHit = intersectRayTriangle2(lRay.getOrigin(),
                                              lRay.getDirection(),
                                              _mesh.vert[lTri.v1],
                                              _mesh.vert[lTri.v2],
                                              _mesh.vert[lTri.v3],
                                              lDistance);

            if (lHit && lDistance < lNearest) {
              lNearest              = lDistance;
              lEtcData[lID].closest = lTri;
            }
          }
        }

        // Backtrac
        while ((lBitStack_lo & 1) == 0) {
          if (lBitStack_lo == 0 && lBitStack_hi == 0) { goto LABEL_END; } // I know, I know...
          lNode        = _nodes[lNode].parent;
          lBitStack_lo = (lBitStack_lo >> 1) | (lBitStack_hi << 63);
          lBitStack_hi >>= 1;
        }

        lNode = _nodes[lNode].isRightChild() ? _nodes[_nodes[lNode].parent].left : _nodes[_nodes[lNode].parent].right;
        lBitStack_lo ^= 1;
      }

    LABEL_END:

      if (lNearest < HUGE_VALF) {
        Triangle lClosest  = lEtcData[lID].closest;
        vec3     lNorm     = triangleNormal(_mesh.vert[lClosest.v1], _mesh.vert[lClosest.v2], _mesh.vert[lClosest.v3]);
        vec3     lHitPos   = lRay.getOrigin() + lNearest * lRay.getDirection();
        vec3     lLightDir = normalize(_light - lHitPos);
        float    lDiffuse  = 1.0f + dot(lNorm, lLightDir);
        lDiffuse           = lDiffuse > 0.0f ? lDiffuse : 0.0f;

        lRes.hit     = 1;
        lRes.diffuse = static_cast<uint8_t>(lDiffuse * 127.0f);
      }

      _img[y * _w + x] = lRes;
    }
  }
}

enum TRAV { TRAV_NONE = 0, TRAV_LEFT = 1, TRAV_RIGHT = 2, TRAV_BOTH = 3 };

extern "C" __device__ __forceinline__ void dReduce64Resolve(int32_t *_res, uint32_t _id) {
  __syncthreads();
  if (_id < 32) {
    _res[_id] += _res[_id + 32];
    _res[_id] += _res[_id + 16];
    _res[_id] += _res[_id + 8];
    _res[_id] += _res[_id + 4];
    _res[_id] += _res[_id + 2];
    _res[_id] += _res[_id + 1];
  }
  __syncthreads();
}

extern "C" __global__ void kTraceRayBundle(Ray *      _rays,
                                           CUDAPixel *_img,
                                           BVHNode *  _nodes,
                                           uint32_t   _rootNode,
                                           MeshRaw    _mesh,
                                           vec3       _light,
                                           uint32_t   _w,
                                           uint32_t   _h) {
  uint32_t iX = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t iY = blockIdx.y * blockDim.y + threadIdx.y;
  uint32_t sX = blockDim.x * gridDim.x;
  uint32_t sY = blockDim.y * gridDim.y;

  uint32_t lID = threadIdx.y * 8 + threadIdx.x;

  for (uint32_t y = iY; y < _h; y += sY) {
    for (uint32_t x = iX; x < _w; x += sX) {
      Ray       lRay = _rays[y * _w + x];
      CUDAPixel lRes = {0, 0, 0};

      __shared__ BVHNode lNodes[3]; // 0: left // 1: right // 2: current
      __shared__ int32_t lResolve[64];
      __shared__ bool    lTravNext[4]; // 0: none // 1: left // 2: right // 3: both
      __shared__ DataShared lEtcData[64];
      __shared__ uint32_t lChildren[2]; // 0: left // 1: right

      uint64_t lBitStack_lo = 0;
      uint64_t lBitStack_hi = 0;
      uint32_t lNode        = _rootNode;

      float lNearest = HUGE_VALF;

      while (true) {
        if (lID == 0) {
          lNodes[2]    = _nodes[lNode];
          lChildren[0] = lNodes[2].left;
          lChildren[1] = lNodes[2].right;
        }
        __syncthreads();

        if (!lNodes[2].isLeaf()) {
          lRes.intCount++;

          if (lID < 2) { lNodes[lID] = _nodes[lChildren[lID]]; } // Load children into shared memory
          __syncthreads();

          float lMinLeft;
          float lMinRight;

          uint32_t lLeftHit  = intersectRayAABB(lNodes[0].bbox, lRay, 0.01f, lNearest + 0.01f, lMinLeft);
          uint32_t lRightHit = intersectRayAABB(lNodes[1].bbox, lRay, 0.01f, lNearest + 0.01f, lMinRight);

          if (lID < 4) { lTravNext[lID] = false; } // Reset trav next
          __syncthreads();
          lTravNext[lRightHit * 2 + lLeftHit] = true;
          __syncthreads();

          // ========================
          // = Check wat to do next =
          // ========================

          if (lTravNext[TRAV_BOTH] || (lTravNext[TRAV_LEFT] && lTravNext[TRAV_RIGHT])) {
            // Both hit somehow
            lBitStack_hi = (lBitStack_hi << 1) | (lBitStack_lo >> 63);
            lBitStack_lo <<= 1;
            lBitStack_lo |= 1;

            // Set what this node wants
            lResolve[lID] = (lLeftHit && (lMinLeft < lMinRight || !lRightHit)) - // Left is prefered --> 1
                            (lRightHit && (lMinRight < lMinLeft || !lLeftHit));  // Right is prefered --> -1

            dReduce64Resolve(lResolve, lID);

            if (lResolve[0] > 0) { // Left is prefered
              lNode = lChildren[0];
            } else { // Right is prefered
              lNode = lChildren[1];
            }

            continue;
          } else if (lTravNext[TRAV_LEFT]) {
            lBitStack_hi = (lBitStack_hi << 1) | (lBitStack_lo >> 63);
            lBitStack_lo <<= 1;
            lNode = lChildren[0];
            continue;
          } else if (lTravNext[TRAV_RIGHT]) {
            lBitStack_hi = (lBitStack_hi << 1) | (lBitStack_lo >> 63);
            lBitStack_lo <<= 1;
            lNode = lChildren[1];
            continue;
          }
        } else {
          for (uint32_t i = 0; i < lChildren[1]; ++i) {
            Triangle lTri = _mesh.faces[lChildren[0] + i];

            float lDistance;

            bool lHit = intersectRayTriangle2(lRay.getOrigin(),
                                              lRay.getDirection(),
                                              _mesh.vert[lTri.v1],
                                              _mesh.vert[lTri.v2],
                                              _mesh.vert[lTri.v3],
                                              lDistance);

            if (lHit && lDistance < lNearest) {
              lNearest              = lDistance;
              lEtcData[lID].closest = lTri;
            }
          }
        }

        // Backtrac
        while ((lBitStack_lo & 1) == 0) {
          if (lBitStack_lo == 0 && lBitStack_hi == 0) { goto LABEL_END; } // I know, I know...
          lNode        = lNodes[2].parent;
          lBitStack_lo = (lBitStack_lo >> 1) | (lBitStack_hi << 63);
          lBitStack_hi >>= 1;

          if (lID == 0) { lNodes[2] = _nodes[lNode]; }
          __syncthreads();
        }

        lNode = lNodes[2].isRightChild() ? _nodes[lNodes[2].parent].left : _nodes[lNodes[2].parent].right;
        lBitStack_lo ^= 1;
      }

    LABEL_END:

      if (lNearest < HUGE_VALF) {
        Triangle lClosest  = lEtcData[lID].closest;
        vec3     lNorm     = triangleNormal(_mesh.vert[lClosest.v1], _mesh.vert[lClosest.v2], _mesh.vert[lClosest.v3]);
        vec3     lHitPos   = lRay.getOrigin() + lNearest * lRay.getDirection();
        vec3     lLightDir = normalize(_light - lHitPos);
        float    lDiffuse  = 1.0f + dot(lNorm, lLightDir);
        lDiffuse           = lDiffuse > 0.0f ? lDiffuse : 0.0f;

        lRes.hit     = 1;
        lRes.diffuse = static_cast<uint8_t>(lDiffuse * 127.0f);
      }

      _img[y * _w + x] = lRes;
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

extern "C" void tracerImage(Ray *      _rays,
                            CUDAPixel *_img,
                            BVHNode *  _nodes,
                            uint32_t   _rootNode,
                            MeshRaw    _mesh,
                            vec3       _light,
                            uint32_t   _w,
                            uint32_t   _h,
                            bool       _bundle) {
  if (!_rays || !_img) { return; }
  dim3 lBlock(8, 8, 1);
  dim3 lGrid((_w + lBlock.x - 1) / lBlock.x, (_h + lBlock.y - 1) / lBlock.y);

  if (_bundle) {
    kTraceRayBundle<<<lGrid, lBlock>>>(_rays, _img, _nodes, _rootNode, _mesh, _light, _w, _h);
  } else {
    kTraceRay<<<lGrid, lBlock>>>(_rays, _img, _nodes, _rootNode, _mesh, _light, _w, _h);
  }
}

extern "C" __global__ void kExtractIntCountFromIMG(CUDAPixel *_devImage, uint16_t *_out, uint32_t _num) {
  for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < _num; i += blockDim.x * gridDim.x) {
    _out[i] = _devImage[i].intCount;
  }
}

extern "C" uint16_t calcIntCountPercentile(CUDAPixel *_devImage, uint32_t _w, uint32_t _h, float _percent) {
  cudaError_t lRes;
  uint16_t    lResult             = UINT16_MAX;
  void *      lCubTempStorage     = nullptr;
  size_t      lCubTempStorageSize = 0;
  uint32_t    lNum                = _w * _h;

  uint16_t *lSortIn  = nullptr;
  uint16_t *lSortOut = nullptr;

  _percent = min(_percent, 0.9999f);
  _percent = max(_percent, 0.0009f);

  CUDA_RUN(cudaMalloc(&lSortIn, lNum * sizeof(uint16_t)));
  CUDA_RUN(cudaMalloc(&lSortOut, lNum * sizeof(uint16_t)));

  kExtractIntCountFromIMG<<<(lNum + 256 - 1) / 256, 256>>>(_devImage, lSortIn, lNum);

  CUDA_RUN(cub::DeviceRadixSort::SortKeys(lCubTempStorage, lCubTempStorageSize, lSortIn, lSortOut, lNum));
  CUDA_RUN(cudaMalloc(&lCubTempStorage, lCubTempStorageSize));
  CUDA_RUN(cub::DeviceRadixSort::SortKeys(lCubTempStorage, lCubTempStorageSize, lSortIn, lSortOut, lNum));
  CUDA_RUN(cudaMemcpy(&lResult, lSortOut + (size_t)((lNum)*_percent), sizeof(uint16_t), cudaMemcpyDeviceToHost));

error:
  cudaFree(lSortIn);
  cudaFree(lSortOut);
  cudaFree(lCubTempStorage);

  return lResult;
}

struct DynNodeSelector {
  uint32_t begin;
  BVHNode *nodes;

  __host__ __device__ __forceinline__ DynNodeSelector(uint32_t _begin, BVHNode *_nodes)
      : begin(_begin), nodes(_nodes) {}

  __device__ __forceinline__ bool operator()(const uint32_t &a) const { return nodes[a].left >= begin; }
};

extern "C" void selectDynamicLeafNodes(GPUWorkingMemory *    _data,
                                       CUDAMemoryBVHPointer *_GPUbvh,
                                       uint32_t *            _out,
                                       uint32_t *            _numOut,
                                       uint32_t              _beginDynamicTriangle) {
  cudaError_t     lRes;
  int *           lNumSelected        = nullptr;
  void *          lCubTempStorage     = nullptr;
  size_t          lCubTempStorageSize = 0;
  DynNodeSelector lSelector(_beginDynamicTriangle, _GPUbvh->nodes);

  CUDA_RUN(cudaMalloc(&lNumSelected, sizeof(int)));

  CUDA_RUN(cub::DeviceSelect::If(
      lCubTempStorage, lCubTempStorageSize, _data->leafNodes, _out, lNumSelected, _data->numLeafNodes, lSelector));

  CUDA_RUN(cudaMalloc(&lCubTempStorage, lCubTempStorageSize));

  CUDA_RUN(cub::DeviceSelect::If(
      lCubTempStorage, lCubTempStorageSize, _data->leafNodes, _out, lNumSelected, _data->numLeafNodes, lSelector));

  CUDA_RUN(cudaMemcpy(_numOut, lNumSelected, sizeof(int), cudaMemcpyDeviceToHost));

error:
  cudaFree(lNumSelected);
  cudaFree(lCubTempStorage);
}

extern "C" __global__ void kRefitLeafBVHTris(
    uint32_t *_nodeIDs, BVHNode *_nodes, Triangle *_faces, vec3 *_vert, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = index; i < _num; i += stride) {
    uint32_t const lID   = _nodeIDs[i];
    BVHNode        lNode = _nodes[lID];
    Triangle const lFace = _faces[lNode.left];
    vec3 const     lV1   = _vert[lFace.v1];
    vec3 const     lV2   = _vert[lFace.v2];
    vec3 const     lV3   = _vert[lFace.v3];

    lNode.bbox.minMax[0] = lV1;
    lNode.bbox.minMax[1] = lV2;

    if (lV2.x < lNode.bbox.minMax[0].x) { lNode.bbox.minMax[0].x = lV2.x; }
    if (lV2.y < lNode.bbox.minMax[0].y) { lNode.bbox.minMax[0].y = lV2.y; }
    if (lV2.z < lNode.bbox.minMax[0].z) { lNode.bbox.minMax[0].z = lV2.z; }
    if (lV3.x < lNode.bbox.minMax[0].x) { lNode.bbox.minMax[0].x = lV3.x; }
    if (lV3.y < lNode.bbox.minMax[0].y) { lNode.bbox.minMax[0].y = lV3.y; }
    if (lV3.z < lNode.bbox.minMax[0].z) { lNode.bbox.minMax[0].z = lV3.z; }

    if (lV1.x > lNode.bbox.minMax[1].x) { lNode.bbox.minMax[1].x = lV1.x; }
    if (lV1.y > lNode.bbox.minMax[1].y) { lNode.bbox.minMax[1].y = lV1.y; }
    if (lV1.z > lNode.bbox.minMax[1].z) { lNode.bbox.minMax[1].z = lV1.z; }
    if (lV3.x > lNode.bbox.minMax[1].x) { lNode.bbox.minMax[1].x = lV3.x; }
    if (lV3.y > lNode.bbox.minMax[1].y) { lNode.bbox.minMax[1].y = lV3.y; }
    if (lV3.z > lNode.bbox.minMax[1].z) { lNode.bbox.minMax[1].z = lV3.z; }

    lNode.bbox.minMax[0] -= FLT_EPSILON;
    lNode.bbox.minMax[1] += FLT_EPSILON;
    _nodes[lID] = lNode;
  }
}

extern "C" void refitDynamicTris(CUDAMemoryBVHPointer *_GPUbvh,
                                 MeshRaw               _cudaMesh,
                                 uint32_t *            _nodesToUpdate,
                                 uint32_t              _numNodes,
                                 uint32_t              _blockSize) {
  uint32_t lBlks = (_numNodes + _blockSize - 1) / _blockSize;
  kRefitLeafBVHTris<<<lBlks, _blockSize>>>(_nodesToUpdate, _GPUbvh->nodes, _cudaMesh.faces, _cudaMesh.vert, _numNodes);
}


extern "C" bool copyToOGLImage(void **_resource, CUDAPixel *_img, uint32_t _w, uint32_t _h) {
  cudaError_t            lRes;
  cudaArray_t            lDevArray;
  cudaGraphicsResource **lResource = reinterpret_cast<cudaGraphicsResource **>(_resource);

  CUDA_RUN(cudaPeekAtLastError());
  CUDA_RUN(cudaGraphicsMapResources(1, lResource, 0));
  CUDA_RUN(cudaGraphicsSubResourceGetMappedArray(&lDevArray, *lResource, 0, 0));

  CUDA_RUN(cudaMemcpyToArray(lDevArray, 0, 0, _img, _w * _h * sizeof(CUDAPixel), cudaMemcpyDeviceToDevice));

  CUDA_RUN(cudaGraphicsUnmapResources(1, lResource, 0));

  return true;

error:
  return false;
}

extern "C" bool copyToOGLBuffer(void **_resource, vec3 *_data, uint32_t _num) {
  cudaError_t            lRes;
  vec3 *                 lBoundData;
  size_t                 lNumBytes;
  cudaGraphicsResource **lResource = reinterpret_cast<cudaGraphicsResource **>(_resource);

  CUDA_RUN(cudaPeekAtLastError());
  CUDA_RUN(cudaGraphicsMapResources(1, lResource, 0));
  CUDA_RUN(cudaGraphicsResourceGetMappedPointer((void **)&lBoundData, &lNumBytes, *lResource));

  CUDA_RUN(cudaMemcpy(lBoundData, _data, _num * sizeof(vec3), cudaMemcpyDeviceToDevice));

  CUDA_RUN(cudaGraphicsUnmapResources(1, lResource, 0));

  return true;

error:
  return false;
}

extern "C" void copyImageToHost(CUDAPixel *_hostPixel, CUDAPixel *_cudaImg, uint32_t _w, uint32_t _h) {
  cudaMemcpy(_hostPixel, _cudaImg, _w * _h * sizeof(CUDAPixel), cudaMemcpyDeviceToHost);
}


extern "C" bool registerOGLImage(void **_resource, uint32_t _image) {
  cudaError_t lRes;
  CUDA_RUN(cudaGraphicsGLRegisterImage(reinterpret_cast<cudaGraphicsResource **>(_resource),
                                       _image,
                                       GL_TEXTURE_2D,
                                       cudaGraphicsRegisterFlagsWriteDiscard));
  return true;

error:
  return false;
}

extern "C" bool registerOGLBuffer(void **_resource, uint32_t _buffer) {
  cudaError_t lRes;
  CUDA_RUN(cudaGraphicsGLRegisterBuffer(
      reinterpret_cast<cudaGraphicsResource **>(_resource), _buffer, cudaGraphicsMapFlagsWriteDiscard));
  return true;

error:
  return false;
}

extern "C" void unregisterOGL(void *_resource) {
  cudaError_t lRes;
  CUDA_RUN(cudaDeviceSynchronize());
  CUDA_RUN(cudaGraphicsUnregisterResource(reinterpret_cast<cudaGraphicsResource *>(_resource)));
error:
  return;
}


extern "C" bool allocateRays(Ray **_rays, uint32_t _num) {
  cudaError_t lRes;
  CUDA_RUN(cudaMalloc(_rays, _num * sizeof(Ray)));
  return true;

error:
  return false;
}

extern "C" bool allocateImage(CUDAPixel **_img, uint32_t _w, uint32_t _h) {
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

extern "C" void freeImage(CUDAPixel *_img) {
  if (_img) { cudaFree(_img); }
}

extern "C" void tracerDoCudaSync() { cudaDeviceSynchronize(); }
extern "C" void initCUDA_GL() {
  cudaError_t lRes;
  CUDA_RUN(cudaGLSetGLDevice(0));
error:
  return;
}
