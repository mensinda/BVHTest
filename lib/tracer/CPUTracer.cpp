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

#include "CPUTracer.hpp"
#include <glm/gtx/intersect.hpp>
#include <glm/gtx/normal.hpp>
#include <ctime>
#include <fstream>
#include <iostream>

using namespace std;
using namespace std::chrono;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::tracer;
using namespace BVHTest::base;

#define USE_RDTSC 1

#if USE_RDTSC
//  Windows
#  ifdef _WIN32

#    include <intrin.h>
uint64_t rdtsc() { return __rdtsc(); }

//  Linux/GCC
#  else

uint64_t rdtsc() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

#  endif
#endif

CPUTracer::~CPUTracer() {}

Pixel CPUTracer::trace(Ray &_ray, Mesh const &_mesh, BVH &_bvh) {
  Pixel lRes = {121, 167, 229, 0, 0};

  /*
   * Algorithm from:
   *
   * Attila T. Áfra and László Szirmay-Kalos. “Stackless Multi-BVH Traversal for CPU,
   * MIC and GPU Ray Tracing”. In: Computer Graphics Forum 33.1 (2014), pp. 129–140.
   * doi: 10.1111/cgf.12259. eprint: https://onlinelibrary.wiley.com/doi/pdf/
   * 10.1111/cgf.12259. url: https://onlinelibrary.wiley.com/doi/abs/10.1111/
   * cgf.12259.
   */

#if USE_RDTSC
  uint64_t lStart = rdtsc();
#else
  timespec lTSStart;
  timespec lTSEnd;

  clock_gettime(CLOCK_MONOTONIC, &lTSStart);
#endif

  __int128_t     lBitStack = 0;
  BVHNode const *lNode     = &_bvh[0];

  Triangle lClosest        = {0, 0, 0};
  float    lNearest        = numeric_limits<float>::infinity();
  vec3     lBarycentricPos = {0.0f, 0.0f, 0.0f};
  vec3     lBarycentricTemp;

  float lMinLeft;
  float lMinRight;
  float lTemp;

  while (true) {
    if (!lNode->isLeaf()) {
      lRes.intCount++;
      BVHNode const *lLeft     = &_bvh[lNode->left];
      BVHNode const *lRight    = &_bvh[lNode->right];
      bool           lLeftHit  = lLeft->bbox.intersect(_ray, 0.01f, lNearest + 0.01f, lMinLeft, lTemp);
      bool           lRightHit = lRight->bbox.intersect(_ray, 0.01f, lNearest + 0.01f, lMinRight, lTemp);

      if (lLeftHit || lRightHit) {
        lBitStack <<= 1;

        if (lLeftHit && lRightHit) {
          lBitStack |= 1;
          lNode = lMinLeft < lMinRight ? lLeft : lRight;
        } else {
          lNode = lLeftHit ? lLeft : lRight;
        }

        continue;
      }
    } else {
      for (uint32_t i = 0; i < lNode->numFaces(); ++i) {
        Triangle const &lTri = _mesh.faces[lNode->beginFaces() + i];

        bool lHit = intersectRayTriangle(_ray.getOrigin(),
                                         _ray.getDirection(),
                                         _mesh.vert[lTri.v1],
                                         _mesh.vert[lTri.v2],
                                         _mesh.vert[lTri.v3],
                                         lBarycentricTemp);

        // See https://github.com/g-truc/glm/issues/6#issuecomment-23149870 for lBaryPos.z usage
        if (lHit && lBarycentricTemp.z < lNearest) {
          lBarycentricPos   = lBarycentricTemp;
          lNearest          = lBarycentricPos.z;
          lBarycentricPos.z = 1.0f - lBarycentricPos.x - lBarycentricPos.y;
          lClosest          = lTri;
        }
      }
    }

    // Backtrac
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0) { goto LABEL_END; } // I know, I know...
      lNode = &_bvh[lNode->parent];
      lBitStack >>= 1;
    }

    lNode = &_bvh.siblingNode(*lNode);
    lBitStack ^= 1;
  }

LABEL_END:

  if (lNearest < numeric_limits<float>::infinity()) {
    vec3  lV1       = _mesh.vert[lClosest.v1];
    vec3  lV2       = _mesh.vert[lClosest.v2];
    vec3  lV3       = _mesh.vert[lClosest.v3];
    vec3  lNorm     = triangleNormal(lV1, lV2, lV3);
    vec3  lHitPos   = lV1 * lBarycentricPos.x + lV2 * lBarycentricPos.y + lV3 * lBarycentricPos.z;
    vec3  lLightDir = normalize(getLightLocation() - lHitPos);
    float lDiffuse  = std::max(1.0f + dot(lNorm, lLightDir), 0.0f);

    lRes.r = static_cast<uint8_t>(lDiffuse * 127.0f);
    lRes.g = static_cast<uint8_t>(lDiffuse * 127.0f);
    lRes.b = static_cast<uint8_t>(lDiffuse * 127.0f);
  }

#if USE_RDTSC
  lRes.rayTime = rdtsc() - lStart;
#else
  clock_gettime(CLOCK_MONOTONIC, &lTSEnd);
  auto lT1     = seconds(lTSStart.tv_sec) + nanoseconds(lTSStart.tv_nsec);
  auto lT2     = seconds(lTSEnd.tv_sec) + nanoseconds(lTSEnd.tv_nsec);
  lRes.rayTime = duration_cast<nanoseconds>(lT2 - lT1).count();
#endif

  return lRes;
}



ErrorCode CPUTracer::runImpl(State &_state) {
  size_t lOffset = _state.work.size();
  _state.work.resize(lOffset + _state.cameras.size());

  // Gen Rays
  for (size_t i = 0; i < _state.cameras.size(); ++i) {
    auto [lWidth, lHeight] = _state.cameras[i]->getResolution();
    _state.work[lOffset + i].img.pixels.resize(lWidth * lHeight);
    _state.work[lOffset + i].rays       = _state.cameras[i]->genRays();
    _state.work[lOffset + i].img.width  = lWidth;
    _state.work[lOffset + i].img.height = lHeight;
    _state.work[lOffset + i].img.name   = _state.name + "_cam_" + to_string(i);
  }

  // Raytrace
  for (size_t i = lOffset; i < _state.work.size(); ++i) {
    auto &lCurr  = _state.work[i];
    auto  lStart = high_resolution_clock::now();

#pragma omp parallel for
    for (size_t j = 0; j < lCurr.rays.size(); ++j) {
      lCurr.img.pixels[j] = trace(lCurr.rays[j], _state.mesh, _state.bvh);
    };

    auto lEnd  = high_resolution_clock::now();
    lCurr.time = duration_cast<nanoseconds>(lEnd - lStart);
  }

  return ErrorCode::OK;
}
