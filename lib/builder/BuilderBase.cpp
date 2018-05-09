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

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;

BuilderBase::~BuilderBase() {}

BuilderBase::ITER BuilderBase::split(ITER _begin, ITER _end, uint32_t) { return _begin + ((_end - _begin) / 2); }

std::vector<TriWithBB> BuilderBase::boundingVolumesFromMesh(Mesh const &_mesh) {
  std::vector<TriWithBB> lRes;

  lRes.resize(_mesh.faces.size());
  for (size_t i = 0; i < _mesh.faces.size(); ++i) {
    vec3 const &v1 = _mesh.vert[_mesh.faces[i].v1];
    vec3 const &v2 = _mesh.vert[_mesh.faces[i].v2];
    vec3 const &v3 = _mesh.vert[_mesh.faces[i].v3];

    lRes[i].tri = _mesh.faces[i];

    lRes[i].bbox.min.x = std::min(std::min(v1.x, v2.x), v3.x) - std::numeric_limits<float>::epsilon();
    lRes[i].bbox.min.y = std::min(std::min(v1.y, v2.y), v3.y) - std::numeric_limits<float>::epsilon();
    lRes[i].bbox.min.z = std::min(std::min(v1.z, v2.z), v3.z) - std::numeric_limits<float>::epsilon();

    lRes[i].bbox.max.x = std::max(std::max(v1.x, v2.x), v3.x) + std::numeric_limits<float>::epsilon();
    lRes[i].bbox.max.y = std::max(std::max(v1.y, v2.y), v3.y) + std::numeric_limits<float>::epsilon();
    lRes[i].bbox.max.z = std::max(std::max(v1.z, v2.z), v3.z) + std::numeric_limits<float>::epsilon();

    lRes[i].centroid = (v1 + v2 + v3) / 3.0f;
  }

  return lRes;
}


BuilderBase::BuildRes BuilderBase::build(BuilderBase::ITER _begin,
                                         BuilderBase::ITER _end,
                                         vector<BVH> &     _bvh,
                                         vector<Triangle> &_tris,
                                         uint32_t          _parent,
                                         uint32_t          _sibling) {
  size_t   lSize    = _end - _begin;
  uint32_t lNewNode = static_cast<uint32_t>(_bvh.size());
  AABB     lNodeBBox;

  vLevel++;

  if (lSize == 0) { throw runtime_error("BuilderBase: WTF! Fix this " + to_string(__LINE__)); }

  if (lSize == 1) {
    lNodeBBox = _begin[0].bbox;
    _bvh.push_back({lNodeBBox,
                    _parent,
                    UINT32_MAX,                          // sibling (single leave --> no sibling)
                    1,                                   // numFaces
                    static_cast<uint32_t>(_tris.size()), // left (leave ==> pointer to begin of faces)
                    UINT32_MAX});

    _tris.emplace_back(_begin[0].tri);
  } else if (lSize == 2) {
    lNodeBBox = _begin[0].bbox;
    lNodeBBox.mergeWith(_begin[1].bbox);

    _tris.emplace_back(_begin[0].tri);
    _tris.emplace_back(_begin[1].tri);

    uint32_t lNewTrisSize = static_cast<uint32_t>(_tris.size());

    _bvh.push_back({
        lNodeBBox,
        _parent,
        _sibling,
        UINT32_MAX,   // numFaces (inner node --> UINT32_MAX)
        lNewNode + 1, // left
        lNewNode + 2  // right
    });

    // Left child
    _bvh.push_back({
        _begin[0].bbox,
        lNewNode,
        lNewNode + 2,     // sibling (Right child)
        1,                // numFaces (inner node --> UINT32_MAX)
        lNewTrisSize - 2, // left (leave ==> pointer to begin of faces)
        UINT32_MAX        // right
    });

    // Right child
    _bvh.push_back({
        _begin[1].bbox,
        lNewNode,
        lNewNode + 1,     // sibling (Left child)
        1,                // numFaces (inner node --> UINT32_MAX)
        lNewTrisSize - 1, // left (leave ==> pointer to begin of faces)
        UINT32_MAX        // right
    });
  } else {
    ITER lSplitAt = split(_begin, _end, vLevel);

    auto &lNode = _bvh.emplace_back(); // Reserve node -- fill content later

    auto [lID1, lBBOX1] = build(_begin, lSplitAt, _bvh, _tris, lNewNode, UINT32_MAX); // Set sibling later
    auto [lID2, lBBOX2] = build(lSplitAt, _end, _bvh, _tris, lNewNode, lID1);

    _bvh[lID1].sibling = lID2; // Fix sibling

    lBBOX1.mergeWith(lBBOX2);
    lNode = {
        lBBOX1,
        _parent,
        _sibling,
        UINT32_MAX, // numFaces (inner node --> UINT32_MAX)
        lID1,       // left
        lID2        // right
    };
  }

  vLevel--;
  return {lNewNode, lNodeBBox};
}
