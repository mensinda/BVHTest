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

#include "BVH.hpp"

namespace BVHTest {
namespace base {

template <size_t NNode, size_t NPath, size_t NAABB>
class alignas(16) BVHPatch final {
 public:
  struct BBox {
    AABB  box;
    float sarea;
  };

  struct NodePair {
    uint32_t first;
    uint32_t second;
  };

 private:
  BVH *    vBVH;
  uint32_t vSize     = 0;
  uint32_t vNumPaths = 0;

  uint32_t vPatch[NNode];
  BVHNode  vNodes[NNode];

  struct AABBPath {
    uint32_t vAABBPath[NAABB];
    BBox     vAABBs[NAABB];
    uint16_t vPathLength = 0;
    uint16_t vSize       = 0;
  };

  AABBPath vPaths[NPath];

 public:
  BVHPatch() = delete;
  CUDA_CALL BVHPatch(BVH *_bvh) : vBVH(_bvh) { clear(); }

  CUDA_CALL uint32_t sibling(uint32_t _node) {
    return get(_node)->isRightChild() ? get(get(_node)->parent)->left : get(get(_node)->parent)->right;
  }

  CUDA_CALL uint32_t sibling(BVHNode const &_node) {
    return _node.isRightChild() ? get(_node.parent)->left : get(_node.parent)->right;
  }

  CUDA_CALL BVHNode &siblingNode(uint32_t _node) { return get(sibling(_node)); }
  CUDA_CALL BVHNode &siblingNode(BVHNode const &_node) { return get(sibling(_node)); }

  CUDA_CALL uint32_t patchIndex(uint32_t _node) const noexcept {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (i >= vSize) { break; }
      if (vPatch[i] == _node) { return i; }
    }
    return UINT32_MAX;
  }

  CUDA_CALL BVHNode *get(uint32_t _node) {
    uint32_t lIndex = patchIndex(_node);
    if (lIndex == UINT32_MAX) { return vBVH->get(_node); }
    assert(lIndex < vSize);
    return &vNodes[lIndex];
  }

  CUDA_CALL BVHNode *getPatchedNode(uint32_t _patchIndex) { return &vNodes[_patchIndex]; }
  CUDA_CALL uint32_t getPatchedNodeIndex(uint32_t _patchIndex) { return vPatch[_patchIndex]; }

  CUDA_CALL BVHNode *rootNode() { return get(vBVH->root()); }
  CUDA_CALL BVHNode *operator[](uint32_t _node) { return get(_node); }

  CUDA_CALL BVHNode *patchNode(uint32_t _node) {
    assert(vSize < NNode);
    vPatch[vSize]  = _node;
    vNodes[vSize]  = *vBVH->get(_node);
    BVHNode *lNode = &vNodes[vSize];
    vSize++;
    return lNode;
  }

  CUDA_CALL void clear() {
    vSize     = 0;
    vNumPaths = 0;
    for (uint32_t i = 0; i < NPath; ++i) {
      vPaths[i].vSize       = 0;
      vPaths[i].vPathLength = 0;
    }
  }

  CUDA_CALL void apply() {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (i >= vSize) { break; }
      *vBVH->get(vPatch[i]) = vNodes[i];
    }
  }


  /*!
   * \brief Fix the AABBs starting at _node in a path to the root
   */
  CUDA_CALL void patchAABBFrom(uint32_t _node) {
    uint32_t lNodeIndex = _node;
    BVHNode *lStart     = get(lNodeIndex);
    BVHNode *lNode      = lStart;

    // Pair: sibling node (= the node we need to fetch) , current Node index
    uint32_t lNumNodes = 0;
    NodePair lNodePairs[NAABB];
    bool     lLastWasLeft = true; // Always remember if we have to fetch the left or right childs bbox

    // Get node index list
    while (true) {
      // Merge with the sibling of the last processed Node
      if (lNumNodes < NAABB) {
        if (lLastWasLeft) {
          lNodePairs[lNumNodes] = {lNode->right, lNodeIndex};
        } else {
          lNodePairs[lNumNodes] = {lNode->left, lNodeIndex};
        }

        lLastWasLeft = lNode->isLeftChild();
      }

      lNumNodes++;

      if (lNodeIndex == root()) { break; } // We processed the root ==> everything is done

      lNodeIndex = lNode->parent;
      lNode      = get(lNodeIndex);
    }

    // Merge the BBox up the tree
    BBox lAABB = getAABB(lStart->left, lNumNodes - 1);
    for (uint32_t i = 0; i < NAABB; ++i) {
      if (i >= lNumNodes) { break; }

      // Merge with the sibling of the last processed Node
      lAABB.box.mergeWith(getAABB(lNodePairs[i].first, lNumNodes - i - 1).box);

      vPaths[vNumPaths].vAABBPath[i] = lNodePairs[i].second;
      vPaths[vNumPaths].vAABBs[i]    = {lAABB.box, lAABB.box.surfaceArea()};
      vPaths[vNumPaths].vSize++;
    }

    vPaths[vNumPaths].vPathLength = lNumNodes;
    vNumPaths++;
  }

  CUDA_CALL BBox getAABB(uint32_t _node, uint32_t _level) {
    for (int32_t i = NPath - 1; i >= 0; --i) {
      uint32_t lIndex = vPaths[i].vPathLength - _level - 1; // May underflow but this is fine (one check less)
      if (lIndex < vPaths[vNumPaths].vSize && vPaths[i].vAABBPath[lIndex] == _node) { return vPaths[i].vAABBs[lIndex]; }
    }
    BVHNode *lNode = get(_node);
    return {lNode->bbox, lNode->surfaceArea};
  }

  CUDA_CALL size_t size() const noexcept { return vSize; }
  CUDA_CALL bool   empty() const noexcept { return vSize == 0; }
  CUDA_CALL uint32_t root() const noexcept { return vBVH->root(); }
};

} // namespace base
} // namespace BVHTest
