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

#include "BruteForceTracer.hpp"
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
inline uint64_t rdtsc() { return __rdtsc(); }

//  Linux/GCC
#  else

inline uint64_t rdtsc() {
  unsigned int lo, hi;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
}

#  endif
#endif

BruteForceTracer::~BruteForceTracer() {}

Pixel BruteForceTracer::trace(Ray &_ray, Mesh const &_mesh, BVH &) {
  Pixel              lRes      = {121, 167, 229, 0, 0};
  static const float lInfinity = numeric_limits<float>::infinity();

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

  Triangle lClosest = {0, 0, 0};
  dvec2    lBarycentricTemp;
  float    lNearest = lInfinity;
  double   lDistance;

  for (size_t i = 0; i < _mesh.faces.size(); ++i) {
    Triangle const &lTri = _mesh.faces[i];

    bool lHit = intersectRayTriangle<double>(static_cast<dvec3 const &>(_ray.getOrigin()),
                                             static_cast<dvec3 const &>(_ray.getDirection()),
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

  if (lNearest < lInfinity) {
    vec3  lNorm     = triangleNormal(_mesh.vert[lClosest.v1], _mesh.vert[lClosest.v2], _mesh.vert[lClosest.v3]);
    vec3  lHitPos   = _ray.getOrigin() + lNearest * _ray.getDirection();
    vec3  lLightDir = normalize(getLightLocation() - lHitPos);
    float lDiffuse  = std::max(1.0f + dot(lNorm, lLightDir), 0.0f);

    lRes.r = lRes.g = lRes.b = static_cast<uint8_t>(lDiffuse * 127.0f);
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



ErrorCode BruteForceTracer::runImpl(State &_state) {
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

#pragma omp parallel for schedule(dynamic, 512)
    for (size_t j = 0; j < lCurr.rays.size(); ++j) {
      lCurr.img.pixels[j] = trace(lCurr.rays[j], _state.mesh, _state.bvh);
    };

    auto lEnd  = high_resolution_clock::now();
    lCurr.time = duration_cast<nanoseconds>(lEnd - lStart);
  }

  return ErrorCode::OK;
}
