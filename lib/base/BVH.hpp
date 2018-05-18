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
#include "Ray.hpp"
#include <vector>

namespace BVHTest::base {

struct Triangle final {
  uint32_t v1;
  uint32_t v2;
  uint32_t v3;
};

struct Mesh final {
  std::vector<glm::vec3> vert;
  std::vector<glm::vec3> norm;
  std::vector<Triangle>  faces;
};

struct AABB {
  glm::vec3 min;
  glm::vec3 max;

#define SIDE(X, Y) d.X *d.Y
  inline float          surfaceArea() const noexcept {
    glm::vec3 d = max - min;
    return 2.0f * (SIDE(x, y) + SIDE(x, z) + SIDE(y, z));
  }
#undef SIDE

  inline void mergeWith(AABB const &_bbox) {
    min.x = std::min(min.x, _bbox.min.x);
    min.y = std::min(min.y, _bbox.min.y);
    min.z = std::min(min.z, _bbox.min.z);
    max.x = std::max(max.x, _bbox.max.x);
    max.y = std::max(max.y, _bbox.max.y);
    max.z = std::max(max.z, _bbox.max.z);
  }

  // Source: http://www.cs.utah.edu/~awilliam/box/
  inline bool intersect(Ray const &_r, float t0, float t1, float &tmin, float &tmax) const {
    glm::vec3 const &lOrigin = _r.getOrigin();
    glm::vec3 const &lInvDir = _r.getInverseDirection();
    Ray::Sign const &lSign   = _r.getSign();

    float tymin, tymax, tzmin, tzmax;

    glm::vec3 bounds[2] = {min, max};

    tmin  = (bounds[lSign.x].x - lOrigin.x) * lInvDir.x;
    tmax  = (bounds[1 - lSign.x].x - lOrigin.x) * lInvDir.x;
    tymin = (bounds[lSign.y].y - lOrigin.y) * lInvDir.y;
    tymax = (bounds[1 - lSign.y].y - lOrigin.y) * lInvDir.y;

    if ((tmin > tymax) || (tymin > tmax)) return false;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    tzmin = (bounds[lSign.z].z - lOrigin.z) * lInvDir.z;
    tzmax = (bounds[1 - lSign.z].z - lOrigin.z) * lInvDir.z;

    if ((tmin > tzmax) || (tzmin > tmax)) return false;
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    return ((tmin < t1) && (tmax > t0));
  }
};

struct TriWithBB {
  Triangle  tri;
  AABB      bbox;
  glm::vec3 centroid;
};

struct alignas(16) BVH {
  AABB     bbox;
  uint32_t parent;
  uint32_t sibling;
  uint32_t numFaces; // Number of triangles (or UINT32_MAX for inner node)
  uint32_t left;     // Left child or index of first triangle when leaf
  uint32_t right;

  inline bool isLeaf() const noexcept { return numFaces != UINT32_MAX; }
};

inline float calcSAH(std::vector<BVH> const &_bvh, float _cInner = 1.2f, float _cLeaf = 1.0f) {
  if (_bvh.empty()) { return 0.0f; }

  float      lSAH  = 0.0f;
  BVH const &lRoot = _bvh[0];

#pragma omp parallel for reduction(+ : lSAH)
  for (size_t i = 1; i < _bvh.size(); ++i) {
    float lCost = _bvh[i].bbox.surfaceArea();
    if (_bvh[i].isLeaf()) {
      lCost *= _cLeaf;
    } else {
      lCost *= _cInner;
    }

    lSAH = lSAH + lCost;
  }

  return (1 / lRoot.bbox.surfaceArea()) * lSAH;
}

} // namespace BVHTest::base
