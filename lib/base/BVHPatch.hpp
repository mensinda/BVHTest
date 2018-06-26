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

const size_t NNode = 10;
const size_t NPath = 2;
const size_t NAABB = 3;

const uint32_t PINDEX_NODE         = 0;
const uint32_t PINDEX_SIBLING      = 1;
const uint32_t PINDEX_PARENT       = 2;
const uint32_t PINDEX_GRAND_PARENT = 3;
const uint32_t PINDEX_1ST_INSERT   = 4;
const uint32_t PINDEX_2ND_INSERT   = 5;
const uint32_t PINDEX_1ST_ROOT     = 6;
const uint32_t PINDEX_1ST_BEST     = 7;
const uint32_t PINDEX_2ND_ROOT     = 8;
const uint32_t PINDEX_2ND_BEST     = 9;

struct alignas(16) MiniPatch final {
  BVHNodePatch vNodes[NNode];
  uint32_t     vPatch[NNode];

  CUDA_CALL void apply(BVH *_bvh) {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (vPatch[i] == UINT32_MAX) { continue; }
      BVHNode *lNode = _bvh->get(vPatch[i]);
      lNode->parent  = vNodes[i].parent;
      lNode->left    = vNodes[i].left;
      lNode->right   = vNodes[i].right;
      lNode->isLeft  = vNodes[i].isLeft;
    }
  }

  CUDA_CALL void clear() {
    for (uint32_t i = 0; i < NNode; ++i) { vPatch[i] = UINT32_MAX; }
  }
};

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
  BVHNodePatch vNodes[NNode];
  uint32_t     vPatch[NNode];
  uint32_t     vNumPaths = 0;

  struct AABBPath {
    uint32_t vAABBPath[NAABB];
    AABB     vAABBs[NAABB];
    uint32_t vPathLength = 0;
  };

  AABBPath vPaths[NPath];

  BVH *vBVH;

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
      if (vPatch[i] == _node) { return i; }
    }
    return UINT32_MAX;
  }

  CUDA_CALL BVHNodePatch get(uint32_t _node) {
    uint32_t lIndex = patchIndex(_node);
    if (lIndex == UINT32_MAX) { return node2PatchedNode(*vBVH->get(_node)); }
    assert(lIndex < NNode);
    return vNodes[lIndex];
  }

  CUDA_CALL BVHNode *getOrig(uint32_t _node) { return vBVH->get(_node); }
  CUDA_CALL BVHNodePatch *getAlreadyPatched(uint32_t _node) { return &vNodes[patchIndex(_node)]; }

  CUDA_CALL BVHNodePatch *getPatchedNode(uint32_t _patchIndex) { return &vNodes[_patchIndex]; }
  CUDA_CALL uint32_t getPatchedNodeIndex(uint32_t _patchIndex) { return vPatch[_patchIndex]; }

  CUDA_CALL BVHNodePatch *patchNode(uint32_t _node, uint32_t _index) {
    assert(_index < NNode);
    vPatch[_index]      = _node;
    vNodes[_index]      = node2PatchedNode(*vBVH->get(_node));
    BVHNodePatch *lNode = &vNodes[_index];
    return lNode;
  }

  CUDA_CALL void clearNode(uint32_t _index) { vPatch[_index] = UINT32_MAX; }

  CUDA_CALL void clearPaths() {
    for (uint32_t i = 0; i < NPath; ++i) { vPaths[i].vPathLength = 0; }
    vNumPaths = 0;
  }

  CUDA_CALL void clear() {
    for (uint32_t i = 0; i < NNode; ++i) { vPatch[i] = UINT32_MAX; }
    clearPaths();
  }

  CUDA_CALL void apply() {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (vPatch[i] == UINT32_MAX) { continue; }
      BVHNode *lNode = vBVH->get(vPatch[i]);
      lNode->parent  = vNodes[i].parent;
      lNode->left    = vNodes[i].left;
      lNode->right   = vNodes[i].right;
      lNode->isLeft  = vNodes[i].isLeft;
    }
  }

  CUDA_CALL void applyOne(uint32_t _index) {
    assert(_index < NNode);
    assert(vPatch[_index] != UINT32_MAX);
    BVHNode *lNode = vBVH->get(vPatch[_index]);
    lNode->parent  = vNodes[_index].parent;
    lNode->left    = vNodes[_index].left;
    lNode->right   = vNodes[_index].right;
    lNode->isLeft  = vNodes[_index].isLeft;
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
      vPaths[vNumPaths].vAABBs[i]    = lAABB.box;
    }

    vPaths[vNumPaths].vPathLength = lNumNodes;
    vNumPaths++;
  }

  CUDA_CALL BBox getAABB(uint32_t _node, uint32_t _level) {
    for (int32_t i = NPath - 1; i >= 0; --i) {
      uint32_t lIndex = vPaths[i].vPathLength - _level - 1; // May underflow but this is fine (one check less)
      if (lIndex < NAABB && vPaths[i].vAABBPath[lIndex] == _node) {
        return {vPaths[i].vAABBs[lIndex], vPaths[i].vAABBs[lIndex].surfaceArea()};
      }
    }
    BVHNode *lNode = vBVH->get(_node);
    return {lNode->bbox, lNode->surfaceArea};
  }

  CUDA_CALL void genMiniPatch(MiniPatch &_out) const noexcept {
    for (uint32_t i = 0; i < NNode; ++i) {
      _out.vNodes[i] = vNodes[i];
      _out.vPatch[i] = vPatch[i];
    }
  }

  CUDA_CALL uint32_t root() const noexcept { return vBVH->root(); }
};

} // namespace base
} // namespace BVHTest
