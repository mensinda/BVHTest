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
#include <utility>

namespace BVHTest::base {

template <size_t NNode, size_t NPath>
class BVHPatch final {
 private:
  BVH *    vBVH;
  uint32_t vSize     = 0;
  uint32_t vNumAABBs = 0;
  uint32_t vRootIndex;

  std::array<uint32_t, NNode> vPatch;
  std::array<BVHNode, NNode>  vNodes;

  std::array<uint32_t, NPath>               vAABBPath;
  std::array<std::pair<AABB, float>, NPath> vAABBs;

 public:
  BVHPatch() = delete;
  BVHPatch(BVH &_bvh) : vBVH(&_bvh) { clear(); }

  inline uint32_t sibling(uint32_t _node) const {
    return getConst(_node).isRightChild() ? getConst(getConst(_node).parent).left
                                          : getConst(getConst(_node).parent).right;
  }

  inline uint32_t sibling(BVHNode const &_node) const {
    return _node.isRightChild() ? getConst(_node.parent).left : getConst(_node.parent).right;
  }

  inline BVHNode &siblingNode(uint32_t _node) { return get(sibling(_node)); }
  inline BVHNode &siblingNode(BVHNode const &_node) { return get(sibling(_node)); }

  inline uint32_t patchIndex(uint32_t _node) const noexcept {
    for (uint32_t i = 0; i < NNode; ++i) {
      if (vPatch[i] == _node) { return i; }
    }
    return UINT32_MAX;
  }

  inline BVHNode &get(uint32_t _node) {
    uint32_t lIndex = patchIndex(_node);
    if (lIndex == UINT32_MAX) { return (*vBVH)[_node]; }
    return vNodes[lIndex];
  }

  inline BVHNode const &getConst(uint32_t _node) const {
    uint32_t lIndex = patchIndex(_node);
    if (lIndex == UINT32_MAX) { return (*vBVH)[_node]; }
    return vNodes[lIndex];
  }

  inline BVHNode &rootNode() { return get(vRootIndex); }
  inline BVHNode &operator[](uint32_t _node) { return get(_node); }

  inline BVHNode &patchNode(uint32_t _node) {
    vPatch[vSize]  = _node;
    vNodes[vSize]  = (*vBVH)[_node];
    BVHNode &lNode = vNodes[vSize];
    vSize++;
    return lNode;
  }

  /*!
   * \brief Fix the AABBs starting at _node in a path to the root
   */
  inline void patchAABBFrom(uint32_t _node) {
    uint32_t lNodeIndex = _node;
    BVHNode *lNode      = &get(lNodeIndex);

    // Merge the BBox up the tree
    AABB lBBox        = get(lNode->left).bbox;
    bool lLastWasLeft = true; // Always remember if we have to fetch the left or right childs bbox
    while (true) {
      // Merge with the sibling of the last processed Node
      if (lLastWasLeft) {
        lBBox.mergeWith(get(lNode->right).bbox);
      } else {
        lBBox.mergeWith(get(lNode->left).bbox);
      }

      vAABBPath[vNumAABBs] = _node;
      vAABBs[vNumAABBs]    = {lBBox, lBBox.surfaceArea()};
      vNumAABBs++;

      if (lNodeIndex == vRootIndex) { return; } // We processed the root ==> everything is done

      lLastWasLeft = lNode->isLeftChild();
      lNodeIndex   = lNode->parent;
      lNode        = &get(lNodeIndex);
    }
  }

  /*!
   * \brief Get the AABB at tree height / level _level of the node _node
   *
   * Stored AABB path is from start to root: [ maxLevel, maxLevel - 1, ..., 1, 0 ]
   */
  inline std::pair<AABB, float> const &getAABB(uint32_t _node, uint32_t _level) {
    if (_level < vNumAABBs) {
      if (vAABBPath[vNumAABBs - _level - 1] == _node) {
        // At level _level the node _node was patched. For this to work node MUST be at tree level _level
        return vAABBs[_level];
      }
    }

    BVHNode &lNode = get(_node);
    return {lNode.bbox, lNode.surfaceArea};
  }

  inline void clear() {
    vSize      = 0;
    vNumAABBs  = 0;
    vRootIndex = vBVH->root();
    for (uint32_t i = 0; i < NNode; ++i) {
      vPatch[i] = UINT32_MAX;
    }
    for (uint32_t i = 0; i < NPath; ++i) {
      vAABBPath[i] = UINT32_MAX;
    }
  }

  inline void apply() {
    assert(vSize == NNode);
    for (uint32_t i = 0; i < NNode; ++i) {
      (*vBVH)[vPatch[i]] = vNodes[i];
    }
    vBVH->setNewRoot(vRootIndex);
    clear();
  }

  inline size_t   size() const noexcept { return vBVH->size(); }
  inline bool     empty() const noexcept { return vBVH->empty(); }
  inline uint32_t root() const noexcept { return vRootIndex; }
  inline uint16_t maxLevel() const noexcept { return vBVH->maxLevel(); }
  inline void     setNewRoot(uint32_t _root) noexcept { vRootIndex = _root; }
};

typedef BVHPatch<10, 128> BVHPatchBittner;

} // namespace BVHTest::base
