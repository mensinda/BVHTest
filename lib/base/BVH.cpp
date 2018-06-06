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

#include "BVH.hpp"

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;

float BVH::calcSAH(float _cInner, float _cLeaf) {
  if (empty()) { return -1.0f; }
  float lSAH = 0.0f;

#pragma omp parallel for reduction(+ : lSAH)
  for (size_t i = 0; i < size(); ++i) {
    if (i == root()) { continue; }

    float lCost = bvh[i].surfaceArea;
    lCost *= (bvh[i].isLeaf() ? _cLeaf : _cInner);
    lSAH = lSAH + lCost;
  }

  return (1.0f / rootNode()->surfaceArea) * lSAH;
}

void BVH::fixLevels() {
  if (empty()) { return; }

  __uint128_t lBitStack = 0;
  uint16_t    lLevel    = 0;
  BVHNode *   lNode     = rootNode();

  vMaxLevel = 0;

  while (true) {
    lNode->level = lLevel;
    if (!lNode->isLeaf()) {
      lBitStack <<= 1;
      lBitStack |= 1;
      lNode = &bvh[lNode->left];
      lLevel++;
      vMaxLevel = std::max(lLevel, vMaxLevel);
      continue;
    }

    // Backtrack
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0) { return; }
      lNode = &bvh[lNode->parent];
      lBitStack >>= 1;
      lLevel--;
    }

    lNode = &bvh[bvh[lNode->parent].right];
    lBitStack ^= 1;
  }
}

void BVH::fixSurfaceAreas() {

#pragma omp parallel for
  for (size_t i = 0; i < size(); ++i) { bvh[i].surfaceArea = bvh[i].bbox.surfaceArea(); }
}
