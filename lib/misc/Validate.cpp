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

#include "BVHTestCfg.hpp"
#include "Validate.hpp"

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::misc;

Validate::~Validate() {}
void Validate::fromJSON(const json &) {}
json Validate::toJSON() const { return json::object(); }

#define PARENT *lBVH.parent(lNode)
#define LEFT *lBVH.left(lNode)
#define RIGHT *lBVH.right(lNode)
#define REQUIRE(a, cnt)                                                                                                \
  if (not(a)) {                                                                                                        \
    cnt++;                                                                                                             \
    lTotalErros++;                                                                                                     \
  }

#define ERROR_IF(val, str)                                                                                             \
  if (val != 0) { getLogger()->error(str, val); }

bool Validate::checkTree(State &_state) {
  auto &lBVH = _state.bvh;
  if (lBVH.empty()) { return true; }

  __uint128_t lBitStack = 0;
  uint16_t    lLevel    = 0;
  uint16_t    lMaxLevel = 0;
  uint32_t    lNode     = lBVH.root();
  uint32_t    lLastNode = lBVH.root();

  uint32_t lTotalErros           = 0;
  uint32_t lLevelErrors          = 0;
  uint32_t lLeftRightErrors      = 0;
  uint32_t lSameChildrenErrors   = 0;
  uint32_t lWrongParentErrors    = 0;
  uint32_t lWrongCildCountErrors = 0;

  vector<uint16_t> lTraversed;
  lTraversed.resize(lBVH.size(), 0);

  REQUIRE(lNode == *lBVH.parent(lNode), lWrongParentErrors); // Loop at root

  while (true) {
    lTraversed[lNode] = 1;
    REQUIRE(*lBVH.level(lNode) == lLevel, lLevelErrors);
    if (lNode != lBVH.root()) {
      if (lBVH.isLeftChild(lNode)) { REQUIRE(*lBVH.left(PARENT) == lNode, lLeftRightErrors); }
      if (lBVH.isRightChild(lNode)) { REQUIRE(*lBVH.right(PARENT) == lNode, lLeftRightErrors); }
    }

    if (!lBVH.isLeaf(lNode)) {
      REQUIRE(LEFT != RIGHT, lSameChildrenErrors);
      REQUIRE(*lBVH.numChildren(lNode) == *lBVH.numChildren(LEFT) + *lBVH.numChildren(RIGHT) + 2,
              lWrongCildCountErrors);
      lBitStack <<= 1;
      lBitStack |= 1;
      lLastNode = lNode;
      lNode     = LEFT;
      REQUIRE(PARENT == lLastNode, lWrongParentErrors);
      lLevel++;
      lMaxLevel = std::max(lLevel, lMaxLevel);
      continue;
    } else {
      REQUIRE(*lBVH.numChildren(lNode) == 0, lWrongCildCountErrors);
    }

    // Backtrack
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0) { goto END_LABEL; } // Yes gotos are evil. We all know that.
      lNode = PARENT;
      lBitStack >>= 1;
      lLevel--;
    }

    lLastNode = PARENT;
    lNode     = *lBVH.right(PARENT);
    REQUIRE(PARENT == lLastNode, lWrongParentErrors);
    lBitStack ^= 1;
  }

END_LABEL:

  uint32_t lNotTraversed = 0;
  for (size_t i = 0; i < lTraversed.size(); ++i) {
    if (lTraversed[i] == 0) { lNotTraversed++; }
  }

  lTotalErros += lNotTraversed;

  ERROR_IF(lLevelErrors, "{:<3} Invalid BVH tree level errors");
  ERROR_IF(lLeftRightErrors, "{:<3} Invalid left / right node indicator");
  ERROR_IF(lSameChildrenErrors, "{:<3} Inner node with the same children");
  ERROR_IF(lWrongParentErrors, "{:<3} Invalid BVH nodes with the wrong parent");
  ERROR_IF(lWrongCildCountErrors, "{:<3} Invalid BVH nodes with a wrong child count");
  ERROR_IF(lNotTraversed, "{:<3} Orphan nodes in the BVH tree");

  if (lMaxLevel != lBVH.maxLevel()) {
    getLogger()->error("Wrong max level: {} but true max level is {}", lBVH.maxLevel(), lMaxLevel);
    lTotalErros++;
  }

  ERROR_IF(lTotalErros, "{:<3} Total BVH tree errors");

  return lTotalErros == 0 ? true : false;
}

bool Validate::checkBBoxes(State &_state) {
  uint32_t lTotalErros = 0;
  BVH &    lBVH        = _state.bvh;

#pragma omp parallel for
  for (uint32_t lNode = 0; lNode < lBVH.size(); ++lNode) {
    if (!lBVH.isLeaf(lNode)) {
      uint32_t lErrors = 0;
      AABB     lBBox   = *lBVH.bbox(lNode);
      AABB     lLBBox  = *lBVH.bbox(LEFT);
      AABB     lRBBox  = *lBVH.bbox(RIGHT);

      REQUIRE(lBBox.min.x <= lLBBox.min.x, lErrors);
      REQUIRE(lBBox.min.y <= lLBBox.min.y, lErrors);
      REQUIRE(lBBox.min.z <= lLBBox.min.z, lErrors);
      REQUIRE(lBBox.max.x >= lLBBox.max.x, lErrors);
      REQUIRE(lBBox.max.y >= lLBBox.max.y, lErrors);
      REQUIRE(lBBox.max.z >= lLBBox.max.z, lErrors);

      REQUIRE(lBBox.min.x <= lRBBox.min.x, lErrors);
      REQUIRE(lBBox.min.y <= lRBBox.min.y, lErrors);
      REQUIRE(lBBox.min.z <= lRBBox.min.z, lErrors);
      REQUIRE(lBBox.max.x >= lRBBox.max.x, lErrors);
      REQUIRE(lBBox.max.y >= lRBBox.max.y, lErrors);
      REQUIRE(lBBox.max.z >= lRBBox.max.z, lErrors);

      if (lErrors != 0) { lTotalErros++; }
    }
  }

  ERROR_IF(lTotalErros, "{:<3} invalid bounding boxes")

  return lTotalErros == 0 ? true : false;
}

bool Validate::checkBBoxesStrict(State &_state) {
  uint32_t lTotalErros = 0;
  BVH &    lBVH        = _state.bvh;

#pragma omp parallel for
  for (uint32_t lNode = 0; lNode < lBVH.size(); ++lNode) {
    if (!lBVH.isLeaf(lNode)) {
      uint32_t lErrors = 0;
      AABB     lBBox   = *lBVH.bbox(lNode);
      AABB     lLBBox  = *lBVH.bbox(LEFT);
      AABB     lRBBox  = *lBVH.bbox(RIGHT);

      REQUIRE(lBBox.min.x == min(lLBBox.min.x, lRBBox.min.x), lErrors);
      REQUIRE(lBBox.min.y == min(lLBBox.min.y, lRBBox.min.y), lErrors);
      REQUIRE(lBBox.min.z == min(lLBBox.min.z, lRBBox.min.z), lErrors);
      REQUIRE(lBBox.max.x == max(lLBBox.max.x, lRBBox.max.x), lErrors);
      REQUIRE(lBBox.max.y == max(lLBBox.max.y, lRBBox.max.y), lErrors);
      REQUIRE(lBBox.max.z == max(lLBBox.max.z, lRBBox.max.z), lErrors)

      if (lErrors != 0) { lTotalErros++; }
    }
  }

  ERROR_IF(lTotalErros, "{:<3} invalid bounding boxes (strict)")

  return lTotalErros == 0 ? true : false;
}

bool Validate::checkSurfaceArea(State &_state) {
  uint32_t lTotalErros = 0;
  BVH &    lBVH        = _state.bvh;

#pragma omp parallel for
  for (uint32_t lNode = 0; lNode < lBVH.size(); ++lNode) {
    REQUIRE(fabs(*lBVH.surfaceArea(lNode) - lBVH.bbox(lNode)->surfaceArea()) < 0.000001f, lTotalErros);
  }

  ERROR_IF(lTotalErros, "{:<3} wrong pre-calculated surface areas");

  return lTotalErros == 0 ? true : false;
}


bool Validate::checkTris(State &_state) {
  uint32_t lTotalErros = 0;
  BVH &    lBVH        = _state.bvh;

  vector<uint16_t> lTraversed;
  lTraversed.resize(_state.mesh.faces.size(), 0);

#pragma omp parallel for
  for (uint32_t lNode = 0; lNode < lBVH.size(); ++lNode) {
    if (lBVH.isLeaf(lNode)) {
      uint32_t lErrors = 0;
      AABB     lBBox   = *lBVH.bbox(lNode);
      uint32_t lStart  = lBVH.beginFaces(lNode);
      uint32_t lEnd    = lStart + lBVH.numFaces(lNode);

      for (uint32_t j = lStart; j < lEnd; ++j) {
        lTraversed[j]           = 1;
        auto [lV1i, lV2i, lV3i] = _state.mesh.faces[j];
        vec3 const &lV1         = _state.mesh.vert[lV1i];
        vec3 const &lV2         = _state.mesh.vert[lV2i];
        vec3 const &lV3         = _state.mesh.vert[lV3i];

        REQUIRE(lBBox.min.x <= lV1.x, lErrors);
        REQUIRE(lBBox.min.y <= lV1.y, lErrors);
        REQUIRE(lBBox.min.z <= lV1.z, lErrors);
        REQUIRE(lBBox.max.x >= lV1.x, lErrors);
        REQUIRE(lBBox.max.y >= lV1.y, lErrors);
        REQUIRE(lBBox.max.z >= lV1.z, lErrors);

        REQUIRE(lBBox.min.x <= lV2.x, lErrors);
        REQUIRE(lBBox.min.y <= lV2.y, lErrors);
        REQUIRE(lBBox.min.z <= lV2.z, lErrors);
        REQUIRE(lBBox.max.x >= lV2.x, lErrors);
        REQUIRE(lBBox.max.y >= lV2.y, lErrors);
        REQUIRE(lBBox.max.z >= lV2.z, lErrors);

        REQUIRE(lBBox.min.x <= lV3.x, lErrors);
        REQUIRE(lBBox.min.y <= lV3.y, lErrors);
        REQUIRE(lBBox.min.z <= lV3.z, lErrors);
        REQUIRE(lBBox.max.x >= lV3.x, lErrors);
        REQUIRE(lBBox.max.y >= lV3.y, lErrors);
        REQUIRE(lBBox.max.z >= lV3.z, lErrors);
      }

      if (lErrors != 0) { lTotalErros++; }
    }
  }

  uint32_t lNotTraversed = 0;
  for (size_t i = 0; i < lTraversed.size(); ++i) {
    if (lTraversed[i] == 0) { lNotTraversed++; }
  }

  lTotalErros += lNotTraversed;

  ERROR_IF(lNotTraversed, "{:<3} Orphan nodes in the BVH tree");
  ERROR_IF(lTotalErros, "{:<3} Total BVH triangle errors");

  return lTotalErros == 0 ? true : false;
}



ErrorCode Validate::runImpl(State &_state) {
  uint32_t lErrors = 0;

  if (!checkTree(_state)) { lErrors++; }
  if (!checkBBoxes(_state)) { lErrors++; }
  if (!checkBBoxesStrict(_state)) { lErrors++; }
  if (!checkSurfaceArea(_state)) { lErrors++; }
  if (!checkTris(_state)) { lErrors++; }

  return lErrors == 0 ? ErrorCode::OK : ErrorCode::WARNING;
}
