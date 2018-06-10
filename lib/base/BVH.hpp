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

#define GLM_FORCE_NO_CTOR_INIT

#ifdef __CUDACC__
#  ifndef CUDA_CALL
#    define CUDA_CALL __host__ __device__ __forceinline__
#  endif
#else
#  ifndef CUDA_CALL
#    define CUDA_CALL inline
#  endif
#endif

#include <glm/vec3.hpp>
#include "Ray.hpp"
#include <vector>

namespace BVHTest {
namespace base {

struct Triangle final {
  uint32_t v1;
  uint32_t v2;
  uint32_t v3;
};

struct Mesh final {
  std::vector<glm::vec3> vert;
  std::vector<glm::vec3> norm;
  std::vector<Triangle>  faces;
};

struct MeshRaw final {
  glm::vec3 *vert     = nullptr; // CUDA device only memory
  glm::vec3 *norm     = nullptr; // CUDA device only memory
  Triangle * faces    = nullptr; // CUDA device only memory
  uint32_t   numVert  = 0;
  uint32_t   numNorm  = 0;
  uint32_t   numFaces = 0;
};

struct AABB {
  glm::vec3 min;
  glm::vec3 max;

  CUDA_CALL float surfaceArea() const noexcept {
    glm::vec3 d = max - min;
    return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
  }

  CUDA_CALL void mergeWith(AABB const &_bbox) {
    min.x = min.x < _bbox.min.x ? min.x : _bbox.min.x;
    min.y = min.y < _bbox.min.y ? min.y : _bbox.min.y;
    min.z = min.z < _bbox.min.z ? min.z : _bbox.min.z;
    max.x = max.x > _bbox.max.x ? max.x : _bbox.max.x;
    max.y = max.y > _bbox.max.y ? max.y : _bbox.max.y;
    max.z = max.z > _bbox.max.z ? max.z : _bbox.max.z;
  }

  // Source: http://www.cs.utah.edu/~awilliam/box/
  CUDA_CALL bool intersect(Ray const &_r, float t0, float t1, float &tmin, float &tmax) const {
    glm::vec3 const &lOrigin = _r.getOrigin();
    glm::vec3 const &lInvDir = _r.getInverseDirection();
    Ray::Sign const &lSign   = _r.getSign();

    float tymin, tymax, tzmin, tzmax;

    glm::vec3 bounds[2] = {min, max};

    tmin  = (bounds[lSign.x].x - lOrigin.x) * lInvDir.x;
    tmax  = (bounds[1 - lSign.x].x - lOrigin.x) * lInvDir.x;
    tymin = (bounds[lSign.y].y - lOrigin.y) * lInvDir.y;
    tymax = (bounds[1 - lSign.y].y - lOrigin.y) * lInvDir.y;

    if ((tmin > tymax) || (tymin > tmax)) return false;
    if (tymin > tmin) tmin = tymin;
    if (tymax < tmax) tmax = tymax;

    tzmin = (bounds[lSign.z].z - lOrigin.z) * lInvDir.z;
    tzmax = (bounds[1 - lSign.z].z - lOrigin.z) * lInvDir.z;

    if ((tmin > tzmax) || (tzmin > tmax)) return false;
    if (tzmin > tmin) tmin = tzmin;
    if (tzmax < tmax) tmax = tzmax;

    return ((tmin < t1) && (tmax > t0));
  }
};

struct TriWithBB {
  Triangle  tri;
  AABB      bbox;
  glm::vec3 centroid;
};

static const uint8_t TRUE  = 1;
static const uint8_t FALSE = 0;

struct alignas(16) BVHNode {
  AABB     bbox;
  uint32_t parent;      // Index of the parent
  uint32_t numChildren; // Total number of children of the node (0 == leaf)
  uint32_t left;        // Left child or index of first triangle when leaf
  uint32_t right;       // Right child or number of faces when leaf

  uint32_t isLeft : 1;          // 1 if the Node is the left child of the parent -- 0 otherwise (right child)
  uint32_t : 7;                 // Force alignment
  uint32_t unused16BitInt : 16; // Find some use for this
  uint32_t level : 8;           // Tree height of the current node

  float surfaceArea; // TODO: Find a better use for these 4 Byte

  CUDA_CALL bool isLeaf() const noexcept { return numChildren == 0; }
  CUDA_CALL uint32_t beginFaces() const noexcept { return left; }
  CUDA_CALL uint32_t numFaces() const noexcept { return right; }
  CUDA_CALL bool     isLeftChild() const noexcept { return isLeft != 0; }
  CUDA_CALL bool     isRightChild() const noexcept { return isLeft == 0; }
};

class BVH final {
 private:
  BVHNode *bvh        = nullptr;
  size_t   vSize      = 0;
  size_t   vCapacity  = 0;
  uint32_t vRootIndex = 0;
  uint16_t vMaxLevel  = 0;

 public:
  CUDA_CALL ~BVH() { free(bvh); }

  CUDA_CALL uint32_t sibling(uint32_t _node) const {
    return bvh[_node].isRightChild() ? bvh[bvh[_node].parent].left : bvh[bvh[_node].parent].right;
  }

  CUDA_CALL uint32_t sibling(BVHNode const *_node) const {
    return _node->isRightChild() ? bvh[_node->parent].left : bvh[_node->parent].right;
  }

  CUDA_CALL BVHNode *siblingNode(uint32_t _node) { return bvh + sibling(_node); }
  CUDA_CALL BVHNode *siblingNode(BVHNode const *_node) { return bvh + sibling(_node); }

  CUDA_CALL size_t size() const noexcept { return vSize; }
  CUDA_CALL bool   empty() const noexcept { return vSize == 0; }
  CUDA_CALL BVHNode *data() { return bvh; }
  CUDA_CALL char *   dataBin() { return reinterpret_cast<char *>(bvh); }
  CUDA_CALL uint32_t root() const noexcept { return vRootIndex; }
  CUDA_CALL BVHNode *rootNode() { return bvh + vRootIndex; }
  CUDA_CALL BVHNode *get(uint32_t _node) { return bvh + _node; }
  CUDA_CALL BVHNode *operator[](uint32_t _node) { return bvh + _node; }
  CUDA_CALL uint32_t nextNodeIndex() const noexcept { return static_cast<uint32_t>(vSize); }
  CUDA_CALL uint16_t maxLevel() const noexcept { return vMaxLevel; }
  CUDA_CALL void     setMaxLevel(uint16_t _level) noexcept { vMaxLevel = _level; }
  CUDA_CALL void     setNewRoot(uint32_t _root) noexcept { vRootIndex = _root; }

  inline void resize(size_t _size) {
    vSize = _size;
    reserve(_size);
  }

  inline void reserve(size_t _size) {
    if (_size > vCapacity) {
      vCapacity = _size;
      bvh       = static_cast<BVHNode *>(realloc(bvh, _size * sizeof(BVHNode)));
    }
  }

  CUDA_CALL void setMemory(BVHNode *_mem, size_t _numNodes, size_t _capacity) {
    bvh       = _mem;
    vSize     = _numNodes;
    vCapacity = _capacity;
  }

  CUDA_CALL uint32_t
            addLeaf(AABB const &_bbox, uint32_t _parent, uint32_t _firstFace, uint32_t _numFaces, bool _isLeft) {
    assert(vSize < vCapacity);

    uint16_t lLevel = empty() ? 0 : bvh[_parent].level + 1;
    vMaxLevel       = lLevel > vMaxLevel ? lLevel : vMaxLevel;
    bvh[vSize]      = {
        _bbox,                  // bbox
        _parent,                // parent
        0,                      // numChildren
        _firstFace,             // left
        _numFaces,              // right
        _isLeft ? TRUE : FALSE, // isLeft
        0,                      // unused16BitInt
        lLevel,                 // treeHeight
        _bbox.surfaceArea()     // surfaceArea
    };
    return vSize++;
  }

  CUDA_CALL uint32_t
            addInner(AABB const &_bbox, uint32_t _parent, uint32_t _numChildren, uint32_t _left, uint32_t _right, bool _isLeft) {
    assert(vSize < vCapacity);

    uint16_t lLevel = empty() ? 0 : bvh[_parent].level + 1;
    vMaxLevel       = lLevel > vMaxLevel ? lLevel : vMaxLevel;
    bvh[vSize]      = {
        _bbox,                  // bbox
        _parent,                // parent
        _numChildren,           // numChildren
        _left,                  // left
        _right,                 // right
        _isLeft ? TRUE : FALSE, // isLeft
        0,                      // unused16BitInt
        lLevel,                 // treeHeight
        _bbox.surfaceArea()     // surfaceArea
    };
    return vSize++;
  }


  float calcSAH(float _cInner = 1.2f, float _cLeaf = 1.0f);
  void  fixLevels();
  void  fixSurfaceAreas();
};

struct CUDAMemoryBVHPointer {
  BVH *    bvh      = nullptr; // CUDA device only memory
  BVHNode *nodes    = nullptr; // CUDA device only memory
  uint32_t numNodes = 0;
};

} // namespace base
} // namespace BVHTest
