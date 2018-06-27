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
void BuilderBase::fromJSON(const json &_j) {
  vCostInner = _j.value("costInner", vCostInner);
  vCostTri   = _j.value("costLeaf", vCostTri);
}

json BuilderBase::toJSON() const { return json{{"costInner", vCostInner}, {"costLeaf", vCostTri}}; }


BuilderBase::ITER BuilderBase::split(ITER _begin, ITER _end, uint32_t) { return _begin + ((_end - _begin) / 2); }

std::vector<TriWithBB> BuilderBase::boundingVolumesFromMesh(Mesh const &_mesh) {
  std::vector<TriWithBB> lRes;

  lRes.resize(_mesh.faces.size());

#pragma omp parallel for
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
                                         BVH &             _bvh,
                                         vector<Triangle> &_tris,
                                         uint32_t          _parent,
                                         bool              _isLeftChild,
                                         uint32_t          _level) {
  size_t   lSize    = _end - _begin;
  uint32_t lNewNode = _bvh.nextNodeIndex();
  AABB     lNodeBBox;

  if (lSize == 0) { throw runtime_error("BuilderBase: WTF! Fix this " + to_string(__LINE__)); }

  if (lSize == 1) {
    lNodeBBox = _begin[0].bbox;
    _bvh.addLeaf(lNodeBBox, _parent, static_cast<uint32_t>(_tris.size()), 1, _isLeftChild);
    _tris.emplace_back(_begin[0].tri);
  } else if (lSize == 2) {
    lNodeBBox = _begin[0].bbox;
    lNodeBBox.mergeWith(_begin[1].bbox);

    _tris.emplace_back(_begin[0].tri);
    _tris.emplace_back(_begin[1].tri);

    uint32_t lNewTrisSize = static_cast<uint32_t>(_tris.size());

    _bvh.addInner(lNodeBBox,    // AABB
                  _parent,      // parent
                  2,            // num children
                  lNewNode + 1, // left
                  lNewNode + 2, // right
                  _isLeftChild  // is current node a left child?
    );

    _bvh.addLeaf(_begin[0].bbox, lNewNode, lNewTrisSize - 2, 1, true);  // Left child
    _bvh.addLeaf(_begin[1].bbox, lNewNode, lNewTrisSize - 1, 1, false); // Right child
  } else {
    ITER lSplitAt = split(_begin, _end, _level);

    // Reserve node -- fill content later
    _bvh.addInner({}, _parent, UINT32_MAX, UINT32_MAX, UINT32_MAX, _isLeftChild);
    auto [lID1, lBBOX1] = build(_begin, lSplitAt, _bvh, _tris, lNewNode, true, _level + 1); // Left
    auto [lID2, lBBOX2] = build(lSplitAt, _end, _bvh, _tris, lNewNode, false, _level + 1);  // Right

    lNodeBBox = lBBOX1;
    lNodeBBox.mergeWith(lBBOX2);
    *_bvh.bbox(lNewNode)        = lNodeBBox;
    *_bvh.numChildren(lNewNode) = 2 + *_bvh.numChildren(lID1) + *_bvh.numChildren(lID2);
    *_bvh.left(lNewNode)        = lID1;
    *_bvh.right(lNewNode)       = lID2;
    *_bvh.surfaceArea(lNewNode) = lNodeBBox.surfaceArea();
  }

  return {lNewNode, lNodeBBox};
}
