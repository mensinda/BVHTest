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

struct alignas(16) BVHNodePatch {
  uint32_t parent;     // Index of the parent
  uint32_t left;       // Left child or index of first triangle when leaf
  uint32_t right;      // Right child or number of faces when leaf
  uint16_t isLeafFlag; // 0 ==> leaf
  uint16_t isLeft;     // 1 if the Node is the left child of the parent -- 0 otherwise (right child)

  CUDA_CALL bool isLeaf() const noexcept { return isLeafFlag == 0; }
  CUDA_CALL uint32_t beginFaces() const noexcept { return left; }
  CUDA_CALL uint32_t numFaces() const noexcept { return right; }
  CUDA_CALL bool     isLeftChild() const noexcept { return isLeft != 0; }
  CUDA_CALL bool     isRightChild() const noexcept { return isLeft == 0; }
};

// template <size_t NNode>
class MiniPatch final {
  BVHNodePatch vNodes[10];
  uint32_t     vPatch[10];
  BVH *        vBVH;
  uint32_t     vSize = 0;
};

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

  uint32_t     vPatch[NNode];
  BVHNodePatch vNodes[NNode];

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
    return get(_node).isRightChild() ? get(get(_node).parent).left : get(get(_node).parent).right;
  }

  CUDA_CALL uint32_t sibling(BVHNode const &_node) {
    return _node.isRightChild() ? get(_node.parent).left : get(_node.parent).right;
  }

  CUDA_CALL uint32_t sibling(BVHNodePatch const &_node) {
    return _node.isRightChild() ? get(_node.parent).left : get(_node.parent).right;
  }

  static CUDA_CALL BVHNodePatch node2PatchedNode(BVHNode const &_n) noexcept {
    BVHNodePatch lRes;
    lRes.parent     = _n.parent;
    lRes.left       = _n.left;
    lRes.right      = _n.right;
    lRes.isLeft     = _n.isLeft;
    lRes.isLeafFlag = static_cast<uint16_t>(_n.numChildren);
    return lRes;
  }

  CUDA_CALL uint32_t patchIndex(uint32_t _node) const noexcept {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (i >= vSize) { break; }
      if (vPatch[i] == _node) { return i; }
    }
    return UINT32_MAX;
  }

  CUDA_CALL BVHNodePatch get(uint32_t _node) {
    uint32_t lIndex = patchIndex(_node);
    if (lIndex == UINT32_MAX) { return node2PatchedNode(*vBVH->get(_node)); }
    assert(lIndex < vSize);
    return vNodes[lIndex];
  }

  CUDA_CALL BVHNode *getOrig(uint32_t _node) { return vBVH->get(_node); }
  CUDA_CALL BVHNodePatch *getAlreadyPatched(uint32_t _node) { return &vNodes[patchIndex(_node)]; }

  CUDA_CALL BVHNodePatch *getPatchedNode(uint32_t _patchIndex) { return &vNodes[_patchIndex]; }
  CUDA_CALL uint32_t getPatchedNodeIndex(uint32_t _patchIndex) { return vPatch[_patchIndex]; }

  CUDA_CALL BVHNodePatch *patchNode(uint32_t _node) {
    assert(vSize < NNode);
    vPatch[vSize]       = _node;
    vNodes[vSize]       = node2PatchedNode(*vBVH->get(_node));
    BVHNodePatch *lNode = &vNodes[vSize];
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
      BVHNode *lNode = vBVH->get(vPatch[i]);
      lNode->parent  = vNodes[i].parent;
      lNode->left    = vNodes[i].left;
      lNode->right   = vNodes[i].right;
      lNode->isLeft  = vNodes[i].isLeft;
    }
  }


  /*!
   * \brief Fix the AABBs starting at _node in a path to the root
   */
  CUDA_CALL void patchAABBFrom(uint32_t _node) {
    uint32_t     lNodeIndex = _node;
    BVHNodePatch lStart     = get(lNodeIndex);
    BVHNodePatch lNode      = lStart;

    // Pair: sibling node (= the node we need to fetch) , current Node index
    uint32_t lNumNodes = 0;
    NodePair lNodePairs[NAABB];
    bool     lLastWasLeft = true; // Always remember if we have to fetch the left or right childs bbox

    // Get node index list
    while (true) {
      // Merge with the sibling of the last processed Node
      if (lNumNodes < NAABB) {
        if (lLastWasLeft) {
          lNodePairs[lNumNodes] = {lNode.right, lNodeIndex};
        } else {
          lNodePairs[lNumNodes] = {lNode.left, lNodeIndex};
        }

        lLastWasLeft = lNode.isLeftChild();
      }

      lNumNodes++;

      if (lNodeIndex == root()) { break; } // We processed the root ==> everything is done

      lNodeIndex = lNode.parent;
      lNode      = get(lNodeIndex);
    }

    // Merge the BBox up the tree
    BBox lAABB = getAABB(lStart.left, lNumNodes - 1);
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
    BVHNode *lNode = vBVH->get(_node);
    return {lNode->bbox, lNode->surfaceArea};
  }

  CUDA_CALL size_t size() const noexcept { return vSize; }
  CUDA_CALL bool   empty() const noexcept { return vSize == 0; }
  CUDA_CALL uint32_t root() const noexcept { return vBVH->root(); }
};

} // namespace base
} // namespace BVHTest
