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
#include <fstream>
#include <iostream>

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::tracer;
using namespace BVHTest::base;

CPUTracer::~CPUTracer() {}

Pixel CPUTracer::trace(Ray &_ray, Mesh const &_mesh, vector<BVH> const &_bvh) {
  Pixel lRes = {121, 167, 229, 0};

  /*
   * Algorithm from:
   *
   * Attila T. Áfra and László Szirmay-Kalos. “Stackless Multi-BVH Traversal for CPU,
   * MIC and GPU Ray Tracing”. In: Computer Graphics Forum 33.1 (2014), pp. 129–140.
   * doi: 10.1111/cgf.12259. eprint: https://onlinelibrary.wiley.com/doi/pdf/
   * 10.1111/cgf.12259. url: https://onlinelibrary.wiley.com/doi/abs/10.1111/
   * cgf.12259.
   */

  uint64_t   lBitStack = 0;
  BVH const *lNode     = &_bvh[0];

  Triangle lClosest        = {0, 0, 0};
  float    lNearest        = numeric_limits<float>::infinity();
  vec3     lBarycentricPos = {0.0f, 0.0f, 0.0f};
  vec3     lBarycentricTemp;

  while (true) {
    if (!lNode->isLeaf()) {
      lRes.intCount++;
      BVH const *lLeft     = &_bvh[lNode->left];
      BVH const *lRight    = &_bvh[lNode->right];
      bool       lLeftHit  = lLeft->bbox.intersect(_ray, 0.01f, 1000.0f);
      bool       lRightHit = lRight->bbox.intersect(_ray, 0.01f, 1000.0f);

      if (lLeftHit || lRightHit) {
        lBitStack <<= 1;

        if (lLeftHit && lRightHit) {
          lBitStack |= 1;
          lNode = lLeft;
        } else if (lLeftHit) {
          lNode = lLeft;
        } else if (lRightHit) {
          lNode = lRight;
        }

        continue;
      }
    } else {
      for (uint32_t i = 0; i < lNode->numFaces; ++i) {
        Triangle const &lTri = _mesh.faces[lNode->left + i];

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

    lNode = &_bvh[lNode->sibling];
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

  return lRes;
}



ErrorCode CPUTracer::runImpl(State &_state) {
  uint32_t lCounter = 0;
  for (auto &i : _state.cameras) {
    auto [lWidth, lHeight] = i->getResolution();
    vector<Ray>   lRays    = i->genRays();
    vector<Pixel> lIMG;
    lIMG.resize(lRays.size());

#pragma omp parallel for
    for (size_t j = 0; j < lRays.size(); ++j) {
      lIMG[j] = trace(lRays[j], _state.mesh, _state.bvh);
    };

    ErrorCode lRes = writeImage(lIMG, lWidth, lHeight, _state.name + "_cam_" + to_string(lCounter++), _state);
    if (lRes != ErrorCode::OK) { return lRes; }
  }

  return ErrorCode::OK;
}
