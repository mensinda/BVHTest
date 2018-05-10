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
#include "camera/Camera.hpp"
#include <glm/gtx/intersect.hpp>
#include <fstream>
#include <iostream>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::tracer;
using namespace BVHTest::base;
using namespace BVHTest::camera;

CPUTracer::~CPUTracer() {}

void CPUTracer::fromJSON(const json &) {}
json CPUTracer::toJSON() const { return json::object(); }

CPUTracer::Pixel CPUTracer::trace(Ray &_ray, Mesh const &_mesh, vector<BVH> const &_bvh) {
  Pixel lRes = {121, 167, 229};

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

  Triangle lClosest;
  float    lNearest = numeric_limits<float>::infinity();

  while (true) {
    if (!lNode->isLeaf()) {
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

        glm::vec3 lBaryPos;
        bool      lHit = glm::intersectRayTriangle(_ray.getOrigin(),
                                              _ray.getDirection(),
                                              _mesh.vert[lTri.v1],
                                              _mesh.vert[lTri.v2],
                                              _mesh.vert[lTri.v3],
                                              lBaryPos);

        // See https://github.com/g-truc/glm/issues/6#issuecomment-23149870 for lBaryPos.z usage
        if (lHit && lBaryPos.z < lNearest) {
          lNearest   = lBaryPos.z;
          lBaryPos.z = 1.0f - lBaryPos.x - lBaryPos.y;
          lClosest   = lTri;

          lRes.r = static_cast<uint8_t>(lBaryPos.x * 255.0f);
          lRes.g = static_cast<uint8_t>(lBaryPos.y * 255.0f);
          lRes.b = static_cast<uint8_t>(lBaryPos.z * 255.0f);
        }
      }
    }

    // Backtrac
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0) { return lRes; }
      lNode = &_bvh[lNode->parent];
      lBitStack >>= 1;
    }

    lNode = &_bvh[lNode->sibling];
    lBitStack ^= 1;
  }

  return lRes;
}



ErrorCode CPUTracer::runImpl(State &_state) {
  Camera lCam;
  auto [lWidth, lHeight] = lCam.getResolution();
  vector<Ray>   lRays    = lCam.genRays();
  vector<Pixel> lIMG;
  lIMG.reserve(lWidth * lHeight);

  for (auto &i : lRays) {
    lIMG.push_back(trace(i, _state.mesh, _state.bvh));
  };

  string lFileName = _state.input + ".ppm";

  fstream lFile(lFileName, lFile.out | lFile.binary);
  lFile << "P6" << endl << lWidth << " " << lHeight << endl << 255 << endl;

  for (auto const &i : lIMG) {
    lFile << i.r << i.g << i.b;
  }

  return ErrorCode::OK;
}
