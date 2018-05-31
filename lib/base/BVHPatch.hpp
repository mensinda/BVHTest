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
#include <array>
#include <tuple>
#include <utility>

namespace BVHTest::base {

template <size_t NNode, size_t NPath, size_t NAABB>
class BVHPatch final {
 private:
  BVH *    vBVH;
  uint32_t vSize     = 0;
  uint32_t vNumPaths = 0;
  uint32_t vRootIndex;

  std::array<uint32_t, NNode> vPatch;
  std::array<BVHNode, NNode>  vNodes;

  struct AABBPath {
    std::array<uint32_t, NAABB>               vAABBPath;
    std::array<std::pair<AABB, float>, NAABB> vAABBs;
    uint32_t                                  vNumAABBs = 0;
  };

  std::array<AABBPath, NPath> vPaths;

 public:
  BVHPatch() = delete;
  BVHPatch(BVH *_bvh) : vBVH(_bvh) { clear(); }

  inline uint32_t sibling(uint32_t _node) {
    return get(_node)->isRightChild() ? get(get(_node)->parent)->left : get(get(_node)->parent)->right;
  }

  inline uint32_t sibling(BVHNode const &_node) {
    return _node.isRightChild() ? get(_node.parent)->left : get(_node.parent)->right;
  }

  inline BVHNode &siblingNode(uint32_t _node) { return get(sibling(_node)); }
  inline BVHNode &siblingNode(BVHNode const &_node) { return get(sibling(_node)); }

  inline uint32_t patchIndex(uint32_t _node) const noexcept {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (i >= vSize) { break; }
      if (vPatch[i] == _node) { return i; }
    }
    return UINT32_MAX;
  }

  inline BVHNode *get(uint32_t _node) {
    uint32_t lIndex = patchIndex(_node);
    if (lIndex == UINT32_MAX) { return &vBVH->get(_node); }
    assert(lIndex < vSize);
    return &vNodes[lIndex];
  }

  inline BVHNode *rootNode() { return get(vRootIndex); }
  inline BVHNode *operator[](uint32_t _node) { return get(_node); }

  inline BVHNode *patchNode(uint32_t _node) {
    assert(vSize < NNode);
    vPatch[vSize]  = _node;
    vNodes[vSize]  = vBVH->get(_node);
    BVHNode *lNode = &vNodes[vSize];
    vSize++;
    return lNode;
  }

  inline void clear() {
    vSize     = 0;
    vNumPaths = 0;
    for (uint32_t i = 0; i < NPath; ++i) {
      vPaths[i].vNumAABBs = 0;
    }
    vRootIndex = vBVH->root();
  }

  inline void apply() {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (i >= vSize) { break; }
      vBVH->get(vPatch[i]) = vNodes[i];
    }
    vBVH->setNewRoot(vRootIndex);
    clear();
  }

  /*!
   * \brief Fix the AABBs starting at _node in a path to the root
   */
  inline void patchAABBFrom(uint32_t _node) {
    uint32_t lNodeIndex = _node;
    BVHNode *lStart     = get(lNodeIndex);
    BVHNode *lNode      = lStart;

    // Pair: sibling node (= the node we need to fetch) , current Node index
    std::array<std::pair<uint32_t, uint32_t>, NAABB> lNodePairs;
    uint32_t                                         lNumNodes = 0;

    bool lLastWasLeft = true; // Always remember if we have to fetch the left or right childs bbox

    // Get node index list
    while (true) {
      // Merge with the sibling of the last processed Node
      if (lLastWasLeft) {
        lNodePairs[lNumNodes] = {lNode->right, lNodeIndex};
      } else {
        lNodePairs[lNumNodes] = {lNode->left, lNodeIndex};
      }

      assert(lNumNodes < NAABB);
      lNumNodes++;
      if (lNodeIndex == root()) { break; } // We processed the root ==> everything is done

      lLastWasLeft = lNode->isLeftChild();
      lNodeIndex   = lNode->parent;
      lNode        = get(lNodeIndex);
    }

    // Merge the BBox up the tree
    auto [lBBox, lSArea] = getAABB(lStart->left, lNumNodes - 1);
    for (uint32_t i = 0; i < lNumNodes; ++i) {
      // Merge with the sibling of the last processed Node
      auto [lNewAABB, lNewSArea] = getAABB(lNodePairs[i].first, lNumNodes - i - 1);
      lBBox.mergeWith(lNewAABB);

      vPaths[vNumPaths].vAABBPath[lNumNodes - i - 1] = lNodePairs[i].second;
      vPaths[vNumPaths].vAABBs[lNumNodes - i - 1]    = {lBBox, lBBox.surfaceArea()};
    }

    vPaths[vNumPaths].vNumAABBs = lNumNodes;
    vNumPaths++;
  }

  inline std::pair<AABB, float> getAABB(uint32_t _node, uint32_t _level) {
    for (int32_t i = NPath - 1; i >= 0; --i) {
      if (_level < vPaths[i].vNumAABBs && vPaths[i].vAABBPath[_level] == _node) {
        // AAA
        return vPaths[i].vAABBs[_level];
      }
    }
    BVHNode *lNode = get(_node);
    return {lNode->bbox, lNode->surfaceArea};
  }

  // Handle case when node is inserted in the stored AABB path
  inline void nodeUpdated(uint32_t _node, uint32_t _level) {
    for (uint32_t i = 0; i < NPath; ++i) {
      if (_level < vPaths[i].vNumAABBs && vPaths[i].vAABBPath[_level] == _node) {
        uint32_t lOld = vNumPaths;
        vNumPaths     = i;
        vPaths[i].vNumAABBs -= _level + 1; // Shift index to the insertion point on the path
        patchAABBFrom(_node);              // Redo the AABB calculation on the path
        vNumPaths = lOld;
      }
    }
  }

  inline size_t   size() const noexcept { return vBVH->size(); }
  inline bool     empty() const noexcept { return vBVH->empty(); }
  inline uint32_t root() const noexcept { return vRootIndex; }
  inline uint16_t maxLevel() const noexcept { return vBVH->maxLevel(); }
  inline void     setNewRoot(uint32_t _root) noexcept { vRootIndex = _root; }
}; // namespace BVHTest::base

typedef BVHPatch<10, 2, 128> BVHPatchBittner;

} // namespace BVHTest::base
