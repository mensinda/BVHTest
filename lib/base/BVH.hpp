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

  inline float surfaceArea() const noexcept {
    glm::vec3 d = max - min;
    return 2 * d.x * d.y + 2 * d.x * d.z + 2 * d.y * d.z;
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
  uint32_t numFaces;
  uint32_t left;
  uint32_t right;

  inline bool isLeaf() const noexcept { return numFaces != 0; }
};

} // namespace BVHTest::base
