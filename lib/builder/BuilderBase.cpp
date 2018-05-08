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

#include "BuilderBase.hpp"
#include <algorithm>

using namespace glm;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;

BuilderBase::~BuilderBase() {}

std::vector<TriWithBB> BuilderBase::boundingVolumesFromMesh(Mesh const &_mesh) {
  std::vector<TriWithBB> lRes;

  lRes.resize(_mesh.faces.size());
  for (size_t i = 0; i < _mesh.faces.size(); ++i) {
    vec3 const &v1 = _mesh.vert[_mesh.faces[i].v1];
    vec3 const &v2 = _mesh.vert[_mesh.faces[i].v2];
    vec3 const &v3 = _mesh.vert[_mesh.faces[i].v3];

    lRes[i].tri = _mesh.faces[i];

    lRes[i].bbox.min.x = std::min(std::min(v1.x, v2.x), v3.x);
    lRes[i].bbox.max.x = std::max(std::max(v1.x, v2.x), v3.x);

    lRes[i].bbox.min.y = std::min(std::min(v1.y, v2.y), v3.y);
    lRes[i].bbox.max.y = std::max(std::max(v1.y, v2.y), v3.y);

    lRes[i].bbox.min.z = std::min(std::min(v1.z, v2.z), v3.z);
    lRes[i].bbox.max.z = std::max(std::max(v1.z, v2.z), v3.z);

    lRes[i].centroid = (lRes[i].bbox.min + lRes[i].bbox.max) / 2.0f;
  }

  return lRes;
}
