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

struct BVHNode {
  AABB *    bbox        = nullptr;
  uint32_t *parent      = nullptr; // Index of the parent
  uint32_t *numChildren = nullptr; // Total number of children of the node (0 == leaf)
  uint32_t *left        = nullptr; // Left child or index of first triangle when leaf
  uint32_t *right       = nullptr; // Right child or number of faces when leaf

  uint8_t * isLeft = nullptr; // 1 if the Node is the left child of the parent -- 0 otherwise (right child)
  uint16_t *level  = nullptr; // Tree height of the current node

  float *surfaceArea = nullptr; // TODO: Find a better use for these 4 Byte

  CUDA_CALL bool isLeaf(uint32_t _n) const noexcept { return numChildren[_n] == 0; }
  CUDA_CALL uint32_t beginFaces(uint32_t _n) const noexcept { return left[_n]; }
  CUDA_CALL uint32_t numFaces(uint32_t _n) const noexcept { return right[_n]; }
  CUDA_CALL bool     isLeftChild(uint32_t _n) const noexcept { return isLeft[_n] != 0; }
  CUDA_CALL bool     isRightChild(uint32_t _n) const noexcept { return isLeft[_n] == 0; }
};

class BVH final {
 private:
  BVHNode  bvh;
  size_t   vSize      = 0;
  size_t   vCapacity  = 0;
  uint32_t vRootIndex = 0;
  uint16_t vMaxLevel  = 0;

 public:
  inline ~BVH() { clear(); }

  CUDA_CALL uint32_t sibling(uint32_t _node) const {
    return bvh.isRightChild(_node) ? bvh.left[bvh.parent[_node]] : bvh.right[bvh.parent[_node]];
  }

  CUDA_CALL AABB &bbox(uint32_t _node) noexcept { return bvh.bbox[_node]; }
  CUDA_CALL uint32_t &parent(uint32_t _node) noexcept { return bvh.parent[_node]; }
  CUDA_CALL uint32_t &numChildren(uint32_t _node) noexcept { return bvh.numChildren[_node]; }
  CUDA_CALL uint32_t &left(uint32_t _node) noexcept { return bvh.left[_node]; }
  CUDA_CALL uint32_t &right(uint32_t _node) noexcept { return bvh.right[_node]; }
  CUDA_CALL uint16_t &level(uint32_t _node) noexcept { return bvh.level[_node]; }
  CUDA_CALL uint8_t &isLeft(uint32_t _node) noexcept { return bvh.isLeft[_node]; }
  CUDA_CALL float &  surfaceArea(uint32_t _node) noexcept { return bvh.surfaceArea[_node]; }

  CUDA_CALL uint32_t beginFaces(uint32_t _node) const noexcept { return bvh.beginFaces(_node); }
  CUDA_CALL uint32_t numFaces(uint32_t _node) const noexcept { return bvh.numFaces(_node); }
  CUDA_CALL bool     isLeftChild(uint32_t _node) const noexcept { return bvh.isLeftChild(_node); }
  CUDA_CALL bool     isRightChild(uint32_t _node) const noexcept { return bvh.isRightChild(_node); }
  CUDA_CALL bool     isLeaf(uint32_t _node) const noexcept { return bvh.isLeaf(_node); }

  CUDA_CALL size_t size() const noexcept { return vSize; }
  CUDA_CALL bool   empty() const noexcept { return vSize == 0; }
  CUDA_CALL BVHNode data() { return bvh; }
  CUDA_CALL void    setData(BVHNode _data) { bvh = _data; }
  CUDA_CALL uint32_t root() const noexcept { return vRootIndex; }
  CUDA_CALL uint32_t nextNodeIndex() const noexcept { return static_cast<uint32_t>(vSize); }
  CUDA_CALL uint16_t maxLevel() const noexcept { return vMaxLevel; }
  CUDA_CALL void     setMaxLevel(uint16_t _level) noexcept { vMaxLevel = _level; }
  CUDA_CALL void     setNewRoot(uint32_t _root) noexcept { vRootIndex = _root; }

  inline void resize(size_t _size) {
    vSize = _size;
    reserve(_size);
  }

  inline void clear() {
    vSize = 0;
    free(bvh.bbox);
    free(bvh.parent);
    free(bvh.numChildren);
    free(bvh.left);
    free(bvh.right);
    free(bvh.isLeft);
    free(bvh.level);
    free(bvh.surfaceArea);

    bvh.bbox        = nullptr;
    bvh.parent      = nullptr;
    bvh.numChildren = nullptr;
    bvh.left        = nullptr;
    bvh.right       = nullptr;
    bvh.isLeft      = nullptr;
    bvh.level       = nullptr;
    bvh.surfaceArea = nullptr;
  }

  inline void reserve(size_t _size) {
    if (_size > vCapacity) {
      vCapacity       = _size;
      bvh.bbox        = static_cast<AABB *>(realloc(bvh.bbox, _size * sizeof(AABB)));
      bvh.parent      = static_cast<uint32_t *>(realloc(bvh.parent, _size * sizeof(uint32_t)));
      bvh.numChildren = static_cast<uint32_t *>(realloc(bvh.numChildren, _size * sizeof(uint32_t)));
      bvh.left        = static_cast<uint32_t *>(realloc(bvh.left, _size * sizeof(uint32_t)));
      bvh.right       = static_cast<uint32_t *>(realloc(bvh.right, _size * sizeof(uint32_t)));
      bvh.isLeft      = static_cast<uint8_t *>(realloc(bvh.isLeft, _size * sizeof(uint8_t)));
      bvh.level       = static_cast<uint16_t *>(realloc(bvh.level, _size * sizeof(uint16_t)));
      bvh.surfaceArea = static_cast<float *>(realloc(bvh.surfaceArea, _size * sizeof(float)));
    }
  }

  inline void setMemory(BVHNode _mem, size_t _numNodes, size_t _capacity) {
    clear();
    bvh       = _mem;
    vSize     = _numNodes;
    vCapacity = _capacity;
  }

  CUDA_CALL uint32_t
            addLeaf(AABB const &_bbox, uint32_t _parent, uint32_t _firstFace, uint32_t _numFaces, bool _isLeft) {
    assert(vSize < vCapacity);

    uint16_t lLevel        = empty() ? 0 : bvh.level[_parent] + 1;
    vMaxLevel              = lLevel > vMaxLevel ? lLevel : vMaxLevel;
    bvh.bbox[vSize]        = _bbox;
    bvh.parent[vSize]      = _parent;
    bvh.numChildren[vSize] = 0;
    bvh.left[vSize]        = _firstFace;
    bvh.right[vSize]       = _numFaces;
    bvh.isLeft[vSize]      = _isLeft ? TRUE : FALSE;
    bvh.level[vSize]       = lLevel;
    bvh.surfaceArea[vSize] = _bbox.surfaceArea();
    return vSize++;
  }

  CUDA_CALL uint32_t
            addInner(AABB const &_bbox, uint32_t _parent, uint32_t _numChildren, uint32_t _left, uint32_t _right, bool _isLeft) {
    assert(vSize < vCapacity);

    uint16_t lLevel        = empty() ? 0 : bvh.level[_parent] + 1;
    vMaxLevel              = lLevel > vMaxLevel ? lLevel : vMaxLevel;
    bvh.bbox[vSize]        = _bbox;
    bvh.parent[vSize]      = _parent;
    bvh.numChildren[vSize] = _numChildren;
    bvh.left[vSize]        = _left;
    bvh.right[vSize]       = _right;
    bvh.isLeft[vSize]      = _isLeft ? TRUE : FALSE;
    bvh.level[vSize]       = lLevel;
    bvh.surfaceArea[vSize] = _bbox.surfaceArea();
    return vSize++;
  }


  float calcSAH(float _cInner = 1.2f, float _cLeaf = 1.0f);
  void  fixLevels();
  void  fixSurfaceAreas();
};

struct CUDAMemoryBVHPointer {
  BVH *    bvh = nullptr; // CUDA device only memory
  BVHNode  nodes;         // CUDA device only memory
  uint32_t numNodes = 0;
};

} // namespace base
} // namespace BVHTest
