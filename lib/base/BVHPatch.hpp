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

const size_t NNode = 10;
const size_t NPath = 2;
const size_t NAABB = 3;

const uint64_t MASK_INDEX   = (1ul << 32ul) - 1ul;
const uint64_t MASK_CONTROL = ~MASK_INDEX;
const uint64_t PATCHED_BIT  = (1ul << 32ul);

struct BVHNodePatch {
  uint32_t parent[NNode];     // Index of the parent
  uint32_t left[NNode];       // Left child or index of first triangle when leaf
  uint32_t right[NNode];      // Right child or number of faces when leaf
  uint8_t  isLeafFlag[NNode]; // 0 ==> leaf
  uint8_t  isLeft[NNode];     // 1 if the Node is the left child of the parent -- 0 otherwise (right child)

  CUDA_CALL bool isLeaf(uint32_t _n) const noexcept { return isLeafFlag[_n] != 0; }
  CUDA_CALL uint32_t beginFaces(uint32_t _n) const noexcept { return left[_n]; }
  CUDA_CALL uint32_t numFaces(uint32_t _n) const noexcept { return right[_n]; }
  CUDA_CALL bool     isLeftChild(uint32_t _n) const noexcept { return isLeft[_n] != 0; }
  CUDA_CALL bool     isRightChild(uint32_t _n) const noexcept { return isLeft[_n] == 0; }
};

const uint16_t PINDEX_GRAND_PARENT = 0;
const uint16_t PINDEX_1ST_ROOT     = 1;
const uint16_t PINDEX_1ST_BEST     = 2;
const uint16_t PINDEX_NODE         = 3;
const uint16_t PINDEX_1ST_INSERT   = 4;
const uint16_t PINDEX_2ND_ROOT     = 5;
const uint16_t PINDEX_2ND_BEST     = 6;
const uint16_t PINDEX_2ND_INSERT   = 7;
const uint16_t PINDEX_SIBLING      = 8;
const uint16_t PINDEX_PARENT       = 9;

const uint16_t PINDEX_SUBSET_END = 4;

struct alignas(16) MiniPatch final {
  BVHNodePatch vNodes;
  uint32_t     vPatch[NNode];

  CUDA_CALL void apply(BVH *_bvh) {
    for (uint16_t i = 0; i < NNode; ++i) {
      if (vPatch[i] == UINT32_MAX) { continue; }
      *_bvh->parent(vPatch[i]) = vNodes.parent[i];
      *_bvh->left(vPatch[i])   = vNodes.left[i];
      *_bvh->right(vPatch[i])  = vNodes.right[i];
      *_bvh->isLeft(vPatch[i]) = vNodes.isLeft[i];
    }
  }

  CUDA_CALL void applyOne(uint16_t _index, BVH *_bvh) {
    assert(_index < NNode);
    assert(vPatch[_index] != UINT32_MAX);
    *_bvh->parent(vPatch[_index]) = vNodes.parent[_index];
    *_bvh->left(vPatch[_index])   = vNodes.left[_index];
    *_bvh->right(vPatch[_index])  = vNodes.right[_index];
    *_bvh->isLeft(vPatch[_index]) = vNodes.isLeft[_index];
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
  BVHNodePatch vNodes;
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

  // Optimized for findNode -- only checks the required nodes
  CUDA_CALL uint32_t patchIndexSubset(uint32_t _node) const noexcept {
    if (vPatch[0] == _node) { return 0; }
    if (vPatch[1] == UINT32_MAX) { return UINT32_MAX; }
    if (vPatch[1] == _node) { return 1; }
    if (vPatch[2] == _node) { return 2; }
    if (vPatch[3] == _node) { return 3; }

    return UINT32_MAX;
  }

  CUDA_CALL uint32_t patchIndex(uint32_t _node) const noexcept {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (vPatch[i] == _node) { return i; }
    }
    return UINT32_MAX;
  }

  CUDA_CALL uint64_t get(uint32_t _node) {
    uint32_t lIndex = patchIndex(_node);
    if (lIndex == UINT32_MAX) { return _node; }
    assert(lIndex < NNode);
    return PATCHED_BIT | lIndex;
  }

  //! \brief only looks at the subset of the patch that is relevant for findNode
  CUDA_CALL uint64_t getSubset(uint32_t _node) {
    uint32_t lIndex = patchIndexSubset(_node);
    if (lIndex == UINT32_MAX) { return _node; }
    assert(lIndex < NNode);
    return PATCHED_BIT | lIndex;
  }

  CUDA_CALL uint32_t *parent(uint64_t _n) noexcept {
    return ((_n & PATCHED_BIT) != 0) ? &vNodes.parent[(uint32_t)_n] : vBVH->parent((uint32_t)_n);
  }
  CUDA_CALL uint32_t *left(uint64_t _n) noexcept {
    return ((_n & PATCHED_BIT) != 0) ? &vNodes.left[(uint32_t)_n] : vBVH->left((uint32_t)_n);
  }
  CUDA_CALL uint32_t *right(uint64_t _n) noexcept {
    return ((_n & PATCHED_BIT) != 0) ? &vNodes.right[(uint32_t)_n] : vBVH->right((uint32_t)_n);
  }
  CUDA_CALL uint8_t *isLeft(uint64_t _n) noexcept {
    return ((_n & PATCHED_BIT) != 0) ? &vNodes.isLeft[(uint32_t)_n] : vBVH->isLeft((uint32_t)_n);
  }

  CUDA_CALL bool isLeaf(uint64_t _n) const noexcept {
    return ((_n & PATCHED_BIT) != 0) ? vNodes.isLeaf((uint32_t)_n) : vBVH->isLeaf((uint32_t)_n);
  }
  CUDA_CALL uint32_t beginFaces(uint64_t _n) const noexcept {
    return ((_n & PATCHED_BIT) != 0) ? vNodes.beginFaces((uint32_t)_n) : vBVH->beginFaces((uint32_t)_n);
  }
  CUDA_CALL uint32_t numFaces(uint64_t _n) const noexcept {
    return ((_n & PATCHED_BIT) != 0) ? vNodes.numFaces((uint32_t)_n) : vBVH->numFaces((uint32_t)_n);
  }
  CUDA_CALL bool isLeftChild(uint64_t _n) const noexcept {
    return ((_n & PATCHED_BIT) != 0) ? vNodes.isLeftChild((uint32_t)_n) : vBVH->isLeftChild((uint32_t)_n);
  }
  CUDA_CALL bool isRightChild(uint64_t _n) const noexcept {
    return ((_n & PATCHED_BIT) != 0) ? vNodes.isRightChild((uint32_t)_n) : vBVH->isRightChild((uint32_t)_n);
  }


  CUDA_CALL uint32_t *patch_parent(uint16_t _n) noexcept { return &vNodes.parent[_n]; }
  CUDA_CALL uint32_t *patch_left(uint16_t _n) noexcept { return &vNodes.left[_n]; }
  CUDA_CALL uint32_t *patch_right(uint16_t _n) noexcept { return &vNodes.right[_n]; }
  CUDA_CALL uint8_t *patch_isLeft(uint16_t _n) noexcept { return &vNodes.isLeft[_n]; }

  CUDA_CALL bool patch_isLeaf(uint16_t _n) const noexcept { return vNodes.isLeaf(_n); }
  CUDA_CALL uint32_t patch_beginFaces(uint16_t _n) const noexcept { return vNodes.beginFaces(_n); }
  CUDA_CALL uint32_t patch_numFaces(uint16_t _n) const noexcept { return vNodes.numFaces(_n); }
  CUDA_CALL bool     patch_isLeftChild(uint16_t _n) const noexcept { return vNodes.isLeftChild(_n); }
  CUDA_CALL bool     patch_isRightChild(uint16_t _n) const noexcept { return vNodes.isRightChild(_n); }


  CUDA_CALL AABB *orig_bbox(uint32_t _node) noexcept { return vBVH->bbox(_node); }
  CUDA_CALL uint32_t *orig_parent(uint32_t _node) noexcept { return vBVH->parent(_node); }
  CUDA_CALL uint32_t *orig_numChildren(uint32_t _node) noexcept { return vBVH->numChildren(_node); }
  CUDA_CALL uint32_t *orig_left(uint32_t _node) noexcept { return vBVH->left(_node); }
  CUDA_CALL uint32_t *orig_right(uint32_t _node) noexcept { return vBVH->right(_node); }
  CUDA_CALL uint16_t *orig_level(uint32_t _node) noexcept { return vBVH->level(_node); }
  CUDA_CALL uint8_t *orig_isLeft(uint32_t _node) noexcept { return vBVH->isLeft(_node); }
  CUDA_CALL float *  orig_surfaceArea(uint32_t _node) noexcept { return vBVH->surfaceArea(_node); }

  CUDA_CALL uint32_t orig_beginFaces(uint32_t _node) const noexcept { return vBVH->beginFaces(_node); }
  CUDA_CALL uint32_t orig_numFaces(uint32_t _node) const noexcept { return vBVH->numFaces(_node); }
  CUDA_CALL bool     orig_isLeftChild(uint32_t _node) const noexcept { return vBVH->isLeftChild(_node); }
  CUDA_CALL bool     orig_isRightChild(uint32_t _node) const noexcept { return vBVH->isRightChild(_node); }
  CUDA_CALL bool     orig_isLeaf(uint32_t _node) const noexcept { return vBVH->isLeaf(_node); }

  CUDA_CALL uint32_t getPatchedNodeIndex(uint32_t _patchIndex) { return vPatch[_patchIndex]; }

  CUDA_CALL uint16_t patchNode(uint32_t _node, uint16_t _index) {
    assert(_index < NNode);
    vPatch[_index]            = _node;
    vNodes.parent[_index]     = *vBVH->parent(_node);
    vNodes.left[_index]       = *vBVH->left(_node);
    vNodes.right[_index]      = *vBVH->right(_node);
    vNodes.isLeafFlag[_index] = vBVH->isLeaf(_node) ? TRUE : FALSE;
    vNodes.isLeft[_index]     = *vBVH->isLeft(_node);
    return _index;
  }

  CUDA_CALL uint16_t movePatch(uint16_t _from, uint16_t _to) {
    vPatch[_to]            = vPatch[_from];
    vPatch[_from]          = UINT32_MAX;
    vNodes.parent[_to]     = vNodes.parent[_from];
    vNodes.left[_to]       = vNodes.left[_from];
    vNodes.right[_to]      = vNodes.right[_from];
    vNodes.isLeafFlag[_to] = vNodes.isLeafFlag[_from];
    vNodes.isLeft[_to]     = vNodes.isLeft[_from];
    return _to;
  }

  //! \brief Resets only one node
  CUDA_CALL void clearNode(uint16_t _index) { vPatch[_index] = UINT32_MAX; }

  //! \brief Only Resets the paths
  CUDA_CALL void clearPaths() {
    for (uint16_t i = 0; i < NPath; ++i) { vPaths[i].vPathLength = 0; }
    vNumPaths = 0;
  }

  CUDA_CALL void clear() {
    for (uint16_t i = 0; i < NNode; ++i) { vPatch[i] = UINT32_MAX; }
    clearPaths();
  }

  //! \brief applys all patches to the BVH
  CUDA_CALL void apply() {
    for (uint16_t i = 0; i < NNode; ++i) {
      if (vPatch[i] == UINT32_MAX) { continue; }
      *vBVH->parent(vPatch[i]) = vNodes.parent[i];
      *vBVH->left(vPatch[i])   = vNodes.left[i];
      *vBVH->right(vPatch[i])  = vNodes.right[i];
      *vBVH->isLeft(vPatch[i]) = vNodes.isLeft[i];
    }
  }

  //! \brief Applys only one Patch to the BVH
  CUDA_CALL void applyOne(uint16_t _index) {
    assert(_index < NNode);
    assert(vPatch[_index] != UINT32_MAX);
    *vBVH->parent(vPatch[_index]) = vNodes.parent[_index];
    *vBVH->left(vPatch[_index])   = vNodes.left[_index];
    *vBVH->right(vPatch[_index])  = vNodes.right[_index];
    *vBVH->isLeft(vPatch[_index]) = vNodes.isLeft[_index];
  }


  /*!
   * \brief Fix the AABBs starting at _node in a path to the root
   */
  CUDA_CALL void patchAABBFrom(uint32_t _node) {
    uint32_t lNodeIndex = _node;
    uint64_t lStart     = getSubset(lNodeIndex);
    uint64_t lNode      = lStart;

    // Pair: sibling node (= the node we need to fetch) , current Node index
    uint32_t lNumNodes = 0;
    NodePair lNodePairs[NAABB];
    bool     lLastWasLeft = true; // Always remember if we have to fetch the left or right childs bbox

    // Get node index list
    while (true) {
      // Merge with the sibling of the last processed Node
      if (lNumNodes < NAABB) {
        if (lLastWasLeft) {
          lNodePairs[lNumNodes] = {*right(lNode), lNodeIndex};
        } else {
          lNodePairs[lNumNodes] = {*left(lNode), lNodeIndex};
        }

        lLastWasLeft = isLeftChild(lNode);
      }

      lNumNodes++;

      if ((uint32_t)lNodeIndex == root()) { break; } // We processed the root ==> everything is done

      lNodeIndex = *parent(lNode);
      lNode      = getSubset(lNodeIndex);
    }

    // Merge the BBox up the tree
    BBox lAABB = getAABB(*left(lStart), lNumNodes - 1);
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
    return {*vBVH->bbox(_node), *vBVH->surfaceArea(_node)};
  }

  CUDA_CALL void genMiniPatch(MiniPatch &_out) const noexcept {
    _out.vNodes = vNodes;
    for (uint16_t i = 0; i < NNode; ++i) { _out.vPatch[i] = vPatch[i]; }
  }

  CUDA_CALL uint32_t root() const noexcept { return vBVH->root(); }
};

} // namespace base
} // namespace BVHTest
