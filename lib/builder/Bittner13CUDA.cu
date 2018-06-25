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

#include "base/BVH.hpp"
#include "base/BVHPatch.hpp"
#include "cuda/cudaFN.hpp"
#include "Bittner13CUDA.hpp"
#include "CUDAHeap.hpp"
#include <cmath>
#include <cub/cub.cuh>
#include <cuda_profiler_api.h>
#include <iostream>

using namespace glm;
using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::cuda;

#define CUDA_RUN(call)                                                                                                 \
  lRes = call;                                                                                                         \
  if (lRes != cudaSuccess) {                                                                                           \
    cout << "CUDA ERROR (" << __FILE__ << ":" << __LINE__ << "): " << cudaGetErrorString(lRes) << endl;                \
    goto error;                                                                                                        \
  }

#define ALLOCATE(ptr, num, type) CUDA_RUN(cudaMalloc(ptr, num * sizeof(type)));
#define FREE(ptr, num)                                                                                                 \
  cudaFree(ptr);                                                                                                       \
  ptr = nullptr;                                                                                                       \
  num = 0;

#define FREE2(ptr)                                                                                                     \
  cudaFree(ptr);                                                                                                       \
  ptr = nullptr;

#define IF_LOCK(N, VAL) if (atomicCAS(_flags + N, 0u, VAL) == 0u)
#define IF_NOT_LOCK(N, VAL) if (atomicCAS(_flags + N, 0u, VAL) != 0u)
#define RELEASE_LOCK(N) atomicExch(_flags + N, 0u);
#define RELEASE_LOCK_S(N, VAL) atomicCAS(_flags + N, VAL, 0u);

struct CUBLeafSelect {
  BVHNode *nodes;

  CUB_RUNTIME_FUNCTION __forceinline__ CUBLeafSelect(BVHNode *_n) : nodes(_n) {}

  __device__ __forceinline__ bool operator()(const uint32_t &a) const { return nodes[a].numChildren == 0; }
};

__global__ void kResetTodoData(uint32_t *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) { _nodes[i] = i; }
}

__global__ void kInitPatches(PATCH *_patches, BVH *_bvh, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) { new (_patches + i) PATCH(_bvh); }
}


/*  ______                                                _            _                     _     */
/*  | ___ \                                              | |          (_)                   | |    */
/*  | |_/ /___ _ __ ___   _____   _____    __ _ _ __   __| |  _ __ ___ _ _ __  ___  ___ _ __| |_   */
/*  |    // _ \ '_ ` _ \ / _ \ \ / / _ \  / _` | '_ \ / _` | | '__/ _ \ | '_ \/ __|/ _ \ '__| __|  */
/*  | |\ \  __/ | | | | | (_) \ V /  __/ | (_| | | | | (_| | | | |  __/ | | | \__ \  __/ |  | |_   */
/*  \_| \_\___|_| |_| |_|\___/ \_/ \___|  \__,_|_| |_|\__,_| |_|  \___|_|_| |_|___/\___|_|   \__|  */
/*                                                                                                 */



struct CUDAHelperStruct {
  uint32_t node;
  float    cost;
  uint32_t level;

  __device__ __forceinline__ bool operator<(CUDAHelperStruct const &_b) const noexcept { return cost > _b.cost; }
};

struct CUDANodeLevel {
  uint32_t node;
  uint32_t level;
};

struct CUDA_RM_RES {
  struct NodePair {
    uint32_t n1;
    uint32_t n2;
  };

  bool     res;
  NodePair toInsert;
  NodePair unused;
  NodePair grandParentAndSibling;
};

struct CUDA_INS_RES {
  bool     res;
  uint32_t best;
  uint32_t root;
};

__device__ CUDANodeLevel findNode1(uint32_t _n, PATCH &_bvh) {
  float             lBestCost      = HUGE_VALF;
  CUDANodeLevel     lBestNodeIndex = {0, 0};
  BVHNode const *   lNode          = _bvh.getOrig(_n);
  AABB const &      lNodeBBox      = lNode->bbox;
  float             lSArea         = lNode->surfaceArea;
  uint32_t          lSize          = 1;
  CUDAHelperStruct  lPQ[CUDA_QUEUE_SIZE];
  CUDAHelperStruct *lBegin = lPQ;

  lPQ[0] = {_bvh.root(), 0.0f, 0};
  while (lSize > 0) {
    CUDAHelperStruct lCurr     = lPQ[0];
    BVHNodePatch     lCurrNode = _bvh.get(lCurr.node);
    auto             lBBox     = _bvh.getAABB(lCurr.node, lCurr.level);
    CUDA_pop_heap(lBegin, lBegin + lSize);
    lSize--;

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    lBBox.box.mergeWith(lNodeBBox);
    float lDirectCost = lBBox.box.surfaceArea();
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = {lCurr.node, lCurr.level};
    }

    float lNewInduced = lTotalCost - lBBox.sarea;
    if ((lNewInduced + lSArea) < lBestCost) {
      if (!lCurrNode.isLeaf()) {
        assert(lSize + 2 < CUDA_QUEUE_SIZE);
        lPQ[lSize + 0] = {lCurrNode.left, lNewInduced, lCurr.level + 1};
        lPQ[lSize + 1] = {lCurrNode.right, lNewInduced, lCurr.level + 1};
        CUDA_push_heap(lBegin, lBegin + lSize + 1);
        CUDA_push_heap(lBegin, lBegin + lSize + 2);
        lSize += 2;
      }
    }
  }

  return lBestNodeIndex;
}


__device__ CUDANodeLevel findNode2(uint32_t _n, PATCH &_bvh) {
  float            lBestCost      = HUGE_VALF;
  CUDANodeLevel    lBestNodeIndex = {0, 0};
  BVHNode const *  lNode          = _bvh.getOrig(_n);
  AABB const &     lNodeBBox      = lNode->bbox;
  float            lSArea         = lNode->surfaceArea;
  float            lMin           = 0.0f;
  float            lMax           = HUGE_VALF;
  uint32_t         lMinIndex      = 0;
  uint32_t         lMaxIndex      = 1;
  CUDAHelperStruct lPQ[CUDA_ALT_QUEUE_SIZE];
  CUDAHelperStruct lCurr;

  // Init
  for (uint32_t i = 0; i < CUDA_ALT_QUEUE_SIZE; ++i) { lPQ[i].cost = HUGE_VALF; }

  lPQ[0] = {_bvh.root(), 0.0f, 0};
  while (lMin < HUGE_VALF) {
    lCurr                  = lPQ[lMinIndex];
    lPQ[lMinIndex].cost    = HUGE_VALF;
    BVHNodePatch lCurrNode = _bvh.get(lCurr.node);
    auto         lBBox     = _bvh.getAABB(lCurr.node, lCurr.level);

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    lBBox.box.mergeWith(lNodeBBox);
    float lDirectCost = lBBox.box.surfaceArea();
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = {lCurr.node, lCurr.level};
    }

    float lNewInduced = lTotalCost - lBBox.sarea;
    if ((lNewInduced + lSArea) < lBestCost && !lCurrNode.isLeaf()) {
      lPQ[lMinIndex] = {lCurrNode.left, lNewInduced, lCurr.level + 1};
      lPQ[lMaxIndex] = {lCurrNode.right, lNewInduced, lCurr.level + 1};
    }

    lMin = HUGE_VALF;
    lMax = 0.0f;
    for (uint32_t i = 0; i < CUDA_ALT_QUEUE_SIZE; ++i) {
      if (lPQ[i].cost < lMin) {
        lMin      = lPQ[i].cost;
        lMinIndex = i;
      }
      if (lPQ[i].cost > lMax) {
        lMax      = lPQ[i].cost;
        lMaxIndex = i;
      }
    }
  }

  return lBestNodeIndex;
}


__device__ CUDA_RM_RES removeNode(uint32_t _node, PATCH &_bvh, uint32_t _lockID) {
  CUDA_RM_RES lFalse = {false, {0, 0}, {0, 0}, {0, 0}};
  if (_bvh.getOrig(_node)->isLeaf() || _node == _bvh.root()) { return lFalse; }

  BVHNodePatch *lNode         = _bvh.patchNode(_node);
  uint32_t      lSiblingIndex = _bvh.sibling(*lNode);
  uint32_t      lParentIndex  = lNode->parent;

  if (lParentIndex == _bvh.root()) { return lFalse; } // Can not remove node with this algorithm

  BVHNodePatch *lSibling          = _bvh.patchNode(lSiblingIndex);
  BVHNodePatch *lParent           = _bvh.patchNode(lParentIndex);
  uint32_t      lGrandParentIndex = lParent->parent;

  BVHNodePatch *lGrandParent = _bvh.patchNode(lGrandParentIndex);

  // FREE LIST:   lNode, lParent
  // INSERT LIST: lLeft, lRight

  // Remove nodes
  if (lParent->isLeftChild()) {
    lGrandParent->left = lSiblingIndex;
    lSibling->isLeft   = TRUE;
    lSibling->parent   = lGrandParentIndex;
  } else {
    lGrandParent->right = lSiblingIndex;
    lSibling->isLeft    = FALSE;
    lSibling->parent    = lGrandParentIndex;
  }

  // update Bounding Boxes (temporary)
  _bvh.patchAABBFrom(lGrandParentIndex);

  if (_bvh.getOrig(lNode->left)->surfaceArea > _bvh.getOrig(lNode->right)->surfaceArea) {
    return {true, {lNode->left, lNode->right}, {_node, lParentIndex}, {lGrandParentIndex, lSiblingIndex}};
  } else {
    return {true, {lNode->right, lNode->left}, {_node, lParentIndex}, {lGrandParentIndex, lSiblingIndex}};
  }
}


__device__ CUDA_INS_RES
           reinsert(uint32_t _node, uint32_t _unused, PATCH &_bvh, bool _update, uint32_t _lockID, bool _altFindNode) {
  CUDANodeLevel lRes = _altFindNode ? findNode2(_node, _bvh) : findNode1(_node, _bvh);
  if (lRes.node == _bvh.root()) { return {false, 0, 0}; }

  uint32_t      lBestPatchIndex = _bvh.patchIndex(lRes.node); // Check if node is already patched
  BVHNodePatch *lBest           = nullptr;

  if (lBestPatchIndex == UINT32_MAX) {
    lBest = _bvh.patchNode(lRes.node);
  } else {
    // Node is already owned by this thread ==> no need to lock it
    lBest = _bvh.getPatchedNode(lBestPatchIndex);
  }

  BVHNodePatch *lNode           = _bvh.patchNode(_node);
  BVHNodePatch *lUnused         = _bvh.getAlreadyPatched(_unused);
  uint32_t      lRootIndex      = lBest->parent;
  uint32_t      lRootPatchIndex = _bvh.patchIndex(lRootIndex);
  BVHNodePatch *lRoot           = nullptr;

  if (lRootPatchIndex == UINT32_MAX) {
    lRoot = _bvh.patchNode(lRootIndex);
  } else {
    lRoot = _bvh.getPatchedNode(lRootPatchIndex);
  }

  // Insert the unused node
  if (lBest->isLeftChild()) {
    lRoot->left     = _unused;
    lUnused->isLeft = TRUE;
  } else {
    lRoot->right    = _unused;
    lUnused->isLeft = FALSE;
  }


  // Insert the other nodes
  lUnused->parent = lRootIndex;
  lUnused->left   = lRes.node;
  lUnused->right  = _node;

  lBest->parent = _unused;
  lBest->isLeft = TRUE;
  lNode->parent = _unused;
  lNode->isLeft = FALSE;

  if (_update) { _bvh.patchAABBFrom(_unused); }

  return {true, lRes.node, lRootIndex};
}




/*  ______ _        _____               */
/*  |  ___(_)      |_   _|              */
/*  | |_   ___  __   | |_ __ ___  ___   */
/*  |  _| | \ \/ /   | | '__/ _ \/ _ \  */
/*  | |   | |>  <    | | | |  __/  __/  */
/*  \_|   |_/_/\_\   \_/_|  \___|\___|  */
/*                                      */
/*                                      */


__global__ void kInitSumMin(uint32_t *_leaf, SumMinCUDA _SMF, BVHNode *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  uint32_t lNode;
  uint32_t lLeft;
  uint32_t lRight;

  for (uint32_t i = index; i < _num; i += stride) {
    lNode            = _leaf[i];
    _SMF.sums[lNode] = _nodes[lNode].surfaceArea;
    _SMF.mins[lNode] = _nodes[lNode].surfaceArea;
    lNode            = _nodes[lNode].parent;

    while (true) {
      uint32_t lOldLock = atomicAdd(&_SMF.flags[lNode], 1);

      // Check if this thread is first. If yes break
      if (lOldLock == 0) { break; }

      lLeft  = _nodes[lNode].left;
      lRight = _nodes[lNode].right;

      _SMF.sums[lNode] = _SMF.sums[lLeft] + _SMF.sums[lRight] + _nodes[lNode].surfaceArea;
      _SMF.mins[lNode] = _SMF.mins[lLeft] < _SMF.mins[lRight] ? _SMF.mins[lLeft] : _SMF.mins[lRight];

      // Check if root
      if (lNode == _nodes[lNode].parent) { break; }
      lNode = _nodes[lNode].parent;
    }
  }
}


__global__ void kFixTree1(uint32_t *_leaf, SumMinCUDA _SMF, BVHNode *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  AABB     lAABB;
  uint32_t lNode;
  uint32_t lLeft;
  uint32_t lRight;
  float    lSArea;

  for (uint32_t i = index; i < _num; i += stride) {
    lNode            = _leaf[i];
    _SMF.sums[lNode] = _nodes[lNode].surfaceArea;
    _SMF.mins[lNode] = _nodes[lNode].surfaceArea;
    lNode            = _nodes[lNode].parent;

    while (true) {
      uint32_t lOldLock = atomicCAS(&_SMF.flags[lNode], 0, 1);

      // Check if this thread is first. If yes break
      if (lOldLock == 0) { break; }

      lLeft  = _nodes[lNode].left;
      lRight = _nodes[lNode].right;
      lAABB  = _nodes[lLeft].bbox;
      lAABB.mergeWith(_nodes[lRight].bbox);
      lSArea = lAABB.surfaceArea();

      _nodes[lNode].bbox        = lAABB;
      _nodes[lNode].surfaceArea = lSArea;
      _nodes[lNode].numChildren = _nodes[lLeft].numChildren + _nodes[lRight].numChildren + 2;
      _SMF.sums[lNode]          = _SMF.sums[lLeft] + _SMF.sums[lRight] + lSArea;
      _SMF.mins[lNode]          = _SMF.mins[lLeft] < _SMF.mins[lRight] ? _SMF.mins[lLeft] : _SMF.mins[lRight];

      // Check if root
      if (lNode == _nodes[lNode].parent) { break; }
      lNode = _nodes[lNode].parent;
    }
  }
}


__global__ void kFixTree3_1(uint32_t *_toFix, SumMinCUDA _SMF, BVHNode *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = index; i < _num; i += stride) {
    uint32_t lNode = _toFix[i];
    if (lNode == UINT32_MAX) { continue; }

    while (true) {
      if (atomicAdd(&_SMF.flags[lNode], 1) != 0) { break; } // Stop when already locked (locked == 1)

      // Check if root
      if (lNode == _nodes[lNode].parent) { break; }
      lNode = _nodes[lNode].parent;
    }
  }
}

__global__ void kFixTree3_2(uint32_t *_toFix, SumMinCUDA _SMF, BVHNode *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  AABB     lAABB;
  uint32_t lNode;
  uint32_t lLeft;
  uint32_t lRight;
  float    lSArea;

  for (uint32_t i = index; i < _num; i += stride) {
    lNode = _toFix[i];
    if (lNode == UINT32_MAX) { continue; }

    while (true) {
      if (atomicSub(&_SMF.flags[lNode], 1) != 1) { break; } // Stop when already locked (locked == 1)

      lLeft  = _nodes[lNode].left;
      lRight = _nodes[lNode].right;
      lAABB  = _nodes[lLeft].bbox;
      lAABB.mergeWith(_nodes[lRight].bbox);
      lSArea = lAABB.surfaceArea();

      _nodes[lNode].bbox        = lAABB;
      _nodes[lNode].surfaceArea = lSArea;
      _nodes[lNode].numChildren = _nodes[lLeft].numChildren + _nodes[lRight].numChildren + 2;
      _SMF.sums[lNode]          = _SMF.sums[lLeft] + _SMF.sums[lRight] + lSArea;
      _SMF.mins[lNode]          = _SMF.mins[lLeft] < _SMF.mins[lRight] ? _SMF.mins[lLeft] : _SMF.mins[lRight];

      // Check if root
      if (lNode == _nodes[lNode].parent) { break; }
      lNode = _nodes[lNode].parent;
    }
  }
}



/*  ___  ___      _         _                        _       */
/*  |  \/  |     (_)       | |                      | |      */
/*  | .  . | __ _ _ _ __   | | _____ _ __ _ __   ___| |___   */
/*  | |\/| |/ _` | | '_ \  | |/ / _ \ '__| '_ \ / _ \ / __|  */
/*  | |  | | (_| | | | | | |   <  __/ |  | | | |  __/ \__ \  */
/*  \_|  |_/\__,_|_|_| |_| |_|\_\___|_|  |_| |_|\___|_|___/  */
/*                                                           */
/*                                                           */

__global__ void kCalcCost(float *_sum, float *_min, BVHNode *_BVHNode, float *_costOut, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) {
    uint32_t lParent      = _BVHNode[i].parent;
    uint32_t lNumChildren = _BVHNode[i].numChildren;
    float    lSA          = _BVHNode[i].surfaceArea;
    bool     lCanRemove   = (lNumChildren != 0) && (i != lParent);

    _costOut[i] = lCanRemove ? ((lSA * lSA * lSA * (float)lNumChildren) / (_sum[i] * _min[i])) : 0.0f;
  }
}



__global__ void kRemoveAndReinsert1(uint32_t *_todoList,
                                    PATCH *   _patches,
                                    uint32_t *_toFix,
                                    bool      _offsetAccess,
                                    uint32_t  _chunk,
                                    uint32_t  _numChunks,
                                    uint32_t  _chunkSize,
                                    bool      _altFindNode) {
  uint32_t index   = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride  = blockDim.x * gridDim.x;
  uint32_t lLockID = index + 1;

  for (int32_t k = index; k < _chunkSize; k += stride) {
    uint32_t     lNodeIndex = _todoList[_offsetAccess ? k * _numChunks + _chunk : _chunk * _chunkSize + k];
    CUDA_RM_RES  lRmRes     = removeNode(lNodeIndex, _patches[k], lLockID);
    CUDA_INS_RES lR1, lR2;

    if (!lRmRes.res) { continue; }

    lR1 = reinsert(lRmRes.toInsert.n1, lRmRes.unused.n1, _patches[k], true, lLockID, _altFindNode);
    lR2 = reinsert(lRmRes.toInsert.n2, lRmRes.unused.n2, _patches[k], false, lLockID, _altFindNode);

    if (!lR1.res || !lR2.res) { continue; }

    _toFix[k * 3 + 0] = lRmRes.grandParentAndSibling.n1;
    _toFix[k * 3 + 1] = lRmRes.unused.n1;
    _toFix[k * 3 + 2] = lRmRes.unused.n2;
  }
}

__global__ void kRemoveAndReinsert2(uint32_t *_todoList,
                                    PATCH *   _patches,
                                    BVH *     _bvh,
                                    uint32_t *_toFix,
                                    bool      _offsetAccess,
                                    uint32_t  _chunk,
                                    uint32_t  _numChunks,
                                    uint32_t  _chunkSize,
                                    bool      _altFindNode) {
  uint32_t index   = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride  = blockDim.x * gridDim.x;
  uint32_t lLockID = index + 1;

  for (int32_t k = index; k < _chunkSize; k += stride) {
    PATCH        lPatch(_bvh);
    uint32_t     lNodeIndex = _todoList[_offsetAccess ? k * _numChunks + _chunk : _chunk * _chunkSize + k];
    CUDA_RM_RES  lRmRes     = removeNode(lNodeIndex, lPatch, lLockID);
    CUDA_INS_RES lR1, lR2;

    if (!lRmRes.res) { continue; }

    lR1 = reinsert(lRmRes.toInsert.n1, lRmRes.unused.n1, lPatch, true, lLockID, _altFindNode);
    lR2 = reinsert(lRmRes.toInsert.n2, lRmRes.unused.n2, lPatch, false, lLockID, _altFindNode);

    if (!lR1.res || !lR2.res) { continue; }

    _toFix[k * 3 + 0] = lRmRes.grandParentAndSibling.n1;
    _toFix[k * 3 + 1] = lRmRes.unused.n1;
    _toFix[k * 3 + 2] = lRmRes.unused.n2;

    _patches[k] = lPatch;
  }
}


__global__ void kCheckConflicts(PATCH *_patches, uint32_t *_flags, uint32_t *_skip, uint32_t *_toFix, uint32_t _num) {
  uint32_t index   = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride  = blockDim.x * gridDim.x;
  uint32_t lLockID = index + 1;

  for (uint32_t k = index; k < _num; k += stride) {
    PATCH lPatch = _patches[k];

    // clang-format off
    switch (lPatch.size()) {
      case 10: IF_NOT_LOCK(lPatch.getPatchedNodeIndex(9), lLockID) { break; } // fallthrough
      case 9:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(8), lLockID) { break; } // fallthrough
      case 8:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(7), lLockID) { break; } // fallthrough
      case 7:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(6), lLockID) { break; } // fallthrough
      case 6:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(5), lLockID) { break; } // fallthrough
      case 5:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(4), lLockID) { break; } // fallthrough
      case 4:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(3), lLockID) { break; } // fallthrough
      case 3:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(2), lLockID) { break; } // fallthrough
      case 2:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(1), lLockID) { break; } // fallthrough
      case 1:  IF_NOT_LOCK(lPatch.getPatchedNodeIndex(0), lLockID) { break; }
               continue; // Everything is locked
      case 0:  break;
    }
    // clang-format on

    // A break in the switch got executed

    // clang-format off
    switch (lPatch.size()) {
      case 10: RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(9), lLockID); // fallthrough
      case 9:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(8), lLockID); // fallthrough
      case 8:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(7), lLockID); // fallthrough
      case 7:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(6), lLockID); // fallthrough
      case 6:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(5), lLockID); // fallthrough
      case 5:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(4), lLockID); // fallthrough
      case 4:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(3), lLockID); // fallthrough
      case 3:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(2), lLockID); // fallthrough
      case 2:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(1), lLockID); // fallthrough
      case 1:  RELEASE_LOCK_S(lPatch.getPatchedNodeIndex(0), lLockID); // fallthrough
      case 0:  break;
    }
    // clang-format on

    _skip[k] += 1;
    _toFix[k * 3 + 0] = UINT32_MAX;
    _toFix[k * 3 + 1] = UINT32_MAX;
    _toFix[k * 3 + 2] = UINT32_MAX;
    _patches[k].clear();
  }
}


__global__ void kApplyPatches(PATCH *_patches, BVH *_bvh, uint32_t *_flags, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t k = index; k < _num; k += stride) {
    if (_patches[k].size() == 0) { continue; }

    for (uint32_t l = 0; l < 10; ++l) {
      if (l >= _patches[k].size()) { break; }
      RELEASE_LOCK(_patches[k].getPatchedNodeIndex(l));
    }

    _patches[k].apply();
    _patches[k].clear();
  }
}



/*    ___  _                  _ _   _                  __                  _   _                   */
/*   / _ \| |                (_) | | |                / _|                | | (_)                  */
/*  / /_\ \ | __ _  ___  _ __ _| |_| |__  _ __ ___   | |_ _   _ _ __   ___| |_ _  ___  _ __  ___   */
/*  |  _  | |/ _` |/ _ \| '__| | __| '_ \| '_ ` _ \  |  _| | | | '_ \ / __| __| |/ _ \| '_ \/ __|  */
/*  | | | | | (_| | (_) | |  | | |_| | | | | | | | | | | | |_| | | | | (__| |_| | (_) | | | \__ \  */
/*  \_| |_/_|\__, |\___/|_|  |_|\__|_| |_|_| |_| |_| |_|  \__,_|_| |_|\___|\__|_|\___/|_| |_|___/  */
/*            __/ |                                                                                */
/*           |___/                                                                                 */


void fixTree1(GPUWorkingMemory *_data, base::CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  uint32_t lNumBlocks = (_data->numLeafNodes + _blockSize - 1) / _blockSize;
  kFixTree1<<<lNumBlocks, _blockSize>>>(_data->leafNodes, _data->sumMin, _GPUbvh->nodes, _data->numLeafNodes);

  cudaMemset(_data->sumMin.flags, 0, _data->sumMin.num * sizeof(uint32_t));
}

void fixTree3(GPUWorkingMemory *_data, BVHTest::base::CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  uint32_t lNumBlocks = (_data->numNodesToFix + _blockSize - 1) / _blockSize;
  kFixTree3_1<<<lNumBlocks, _blockSize>>>(_data->nodesToFix, _data->sumMin, _GPUbvh->nodes, _data->numNodesToFix);
  kFixTree3_2<<<lNumBlocks, _blockSize>>>(_data->nodesToFix, _data->sumMin, _GPUbvh->nodes, _data->numNodesToFix);
}



void doAlgorithmStep(GPUWorkingMemory *    _data,
                     CUDAMemoryBVHPointer *_GPUbvh,
                     uint32_t              _numChunks,
                     uint32_t              _chunkSize,
                     uint32_t              _blockSize,
                     bool                  _offsetAccess,
                     bool                  _altFindNode,
                     bool                  _altFixTree,
                     bool                  _localPatchCPY) {
  if (!_data || !_GPUbvh) { return; }

  cudaError_t lRes;
  uint32_t    lNumBlocksAll   = (_data->sumMin.num + _blockSize - 1) / _blockSize;
  uint32_t    lNumBlocksChunk = (_chunkSize + _blockSize - 1) / _blockSize;

  kCalcCost<<<lNumBlocksAll, _blockSize>>>(
      _data->sumMin.sums, _data->sumMin.mins, _GPUbvh->nodes, _data->todoNodes.costs, _data->sumMin.num);

  CUDA_RUN(cub::DeviceRadixSort::SortPairsDescending(_data->cubSortTempStorage,
                                                     _data->cubSortTempStorageSize,
                                                     _data->todoNodes.costs,
                                                     _data->todoSorted.costs,
                                                     _data->todoNodes.nodes,
                                                     _data->todoSorted.nodes,
                                                     _data->todoNodes.num));

  for (uint32_t i = 0; i < _numChunks; ++i) {
    if (_localPatchCPY) {
      kRemoveAndReinsert2<<<lNumBlocksChunk, _blockSize>>>(_data->todoSorted.nodes,
                                                           _data->patches,
                                                           _GPUbvh->bvh,
                                                           _data->nodesToFix,
                                                           _offsetAccess,
                                                           i,
                                                           _numChunks,
                                                           _chunkSize,
                                                           _altFindNode);
    } else {
      kRemoveAndReinsert1<<<lNumBlocksChunk, _blockSize>>>(_data->todoSorted.nodes,
                                                           _data->patches,
                                                           _data->nodesToFix,
                                                           _offsetAccess,
                                                           i,
                                                           _numChunks,
                                                           _chunkSize,
                                                           _altFindNode);
    }

    kCheckConflicts<<<lNumBlocksChunk, _blockSize>>>(
        _data->patches, _data->sumMin.flags, _data->skipped, _data->nodesToFix, _chunkSize);

    kApplyPatches<<<lNumBlocksChunk, _blockSize>>>(_data->patches, _GPUbvh->bvh, _data->sumMin.flags, _chunkSize);

    if (_altFixTree) {
      fixTree3(_data, _GPUbvh, _blockSize);
    } else {
      fixTree1(_data, _GPUbvh, _blockSize);
    }
  }

error:
  return;
}


uint32_t calcNumSkipped(GPUWorkingMemory *_data) {
  cudaError_t lRes;
  uint32_t    lSkipped    = 0;
  uint32_t *  lDevSkipped = nullptr;

  ALLOCATE(&lDevSkipped, 1, uint32_t);

  CUDA_RUN(cub::DeviceReduce::Sum(
      _data->cubSortTempStorage, _data->cubSortTempStorageSize, _data->skipped, lDevSkipped, _data->numSkipped));

  CUDA_RUN(cudaMemcpy(&lSkipped, lDevSkipped, sizeof(uint32_t), cudaMemcpyDeviceToHost));

error:
  FREE2(lDevSkipped);
  return lSkipped;
}


void doCudaDevSync() { cudaDeviceSynchronize(); }

/*  ___  ___                                                                                              _     */
/*  |  \/  |                                                                                             | |    */
/*  | .  . | ___ _ __ ___   ___  _ __ _   _   _ __ ___   __ _ _ __   __ _  __ _  ___ _ __ ___   ___ _ __ | |_   */
/*  | |\/| |/ _ \ '_ ` _ \ / _ \| '__| | | | | '_ ` _ \ / _` | '_ \ / _` |/ _` |/ _ \ '_ ` _ \ / _ \ '_ \| __|  */
/*  | |  | |  __/ | | | | | (_) | |  | |_| | | | | | | | (_| | | | | (_| | (_| |  __/ | | | | |  __/ | | | |_   */
/*  \_|  |_/\___|_| |_| |_|\___/|_|   \__, | |_| |_| |_|\__,_|_| |_|\__,_|\__, |\___|_| |_| |_|\___|_| |_|\__|  */
/*                                     __/ |                               __/ |                                */
/*                                    |___/                               |___/                                 */


GPUWorkingMemory allocateMemory(CUDAMemoryBVHPointer *_bvh, uint32_t _batchSize, uint32_t _numFaces) {
  cudaProfilerStart();
  GPUWorkingMemory lMem;

  lMem.result = true;
  cudaError_t lRes;
  size_t      lCubTempStorage1 = 0;
  size_t      lCubTempStorage2 = 0;
  uint32_t *  lTemp            = nullptr;

  lMem.sumMin.num     = _bvh->numNodes;
  lMem.todoNodes.num  = _bvh->numNodes;
  lMem.todoSorted.num = _bvh->numNodes;
  lMem.numLeafNodes   = _numFaces;
  lMem.numPatches     = _batchSize;
  lMem.numSkipped     = _batchSize;
  lMem.numNodesToFix  = _batchSize * 3;

  ALLOCATE(&lMem.sumMin.sums, lMem.sumMin.num, float);
  ALLOCATE(&lMem.sumMin.mins, lMem.sumMin.num, float);
  ALLOCATE(&lMem.sumMin.flags, lMem.sumMin.num, uint32_t);
  ALLOCATE(&lMem.todoNodes.nodes, lMem.todoNodes.num, uint32_t);
  ALLOCATE(&lMem.todoNodes.costs, lMem.todoNodes.num, float);
  ALLOCATE(&lMem.todoSorted.nodes, lMem.todoSorted.num, uint32_t);
  ALLOCATE(&lMem.todoSorted.costs, lMem.todoSorted.num, float);
  ALLOCATE(&lMem.leafNodes, lMem.numLeafNodes, uint32_t);
  ALLOCATE(&lMem.patches, lMem.numPatches, PATCH);
  ALLOCATE(&lMem.skipped, lMem.numSkipped, uint32_t);
  ALLOCATE(&lMem.nodesToFix, lMem.numNodesToFix, uint32_t);



  // This only calculates the memory requirements
  CUDA_RUN(cub::DeviceRadixSort::SortPairsDescending(lMem.cubSortTempStorage,
                                                     lCubTempStorage1,
                                                     lMem.todoNodes.costs,
                                                     lMem.todoSorted.costs,
                                                     lMem.todoNodes.nodes,
                                                     lMem.todoSorted.nodes,
                                                     lMem.todoNodes.num));

  CUDA_RUN(cub::DeviceReduce::Sum(lMem.cubSortTempStorage, lCubTempStorage2, lMem.skipped, lTemp, lMem.numSkipped));

  lMem.cubSortTempStorageSize = lCubTempStorage1 > lCubTempStorage2 ? lCubTempStorage1 : lCubTempStorage2;

  ALLOCATE(&lMem.cubSortTempStorage, lMem.cubSortTempStorageSize, uint8_t);

  return lMem;

error:
  lMem.result = false;

  FREE(lMem.sumMin.sums, lMem.sumMin.num);
  FREE(lMem.sumMin.mins, lMem.sumMin.num);
  FREE(lMem.sumMin.flags, lMem.sumMin.num);
  FREE(lMem.todoNodes.nodes, lMem.todoNodes.num);
  FREE(lMem.todoNodes.costs, lMem.todoNodes.num);
  FREE(lMem.todoSorted.nodes, lMem.todoNodes.num);
  FREE(lMem.todoSorted.costs, lMem.todoNodes.num);
  FREE(lMem.leafNodes, lMem.numLeafNodes);
  FREE(lMem.patches, lMem.numPatches);
  FREE(lMem.skipped, lMem.numSkipped);
  FREE(lMem.nodesToFix, lMem.numNodesToFix);
  FREE(lMem.cubSortTempStorage, lMem.cubSortTempStorageSize);

  return lMem;
}

void freeMemory(GPUWorkingMemory *_data) {
  _data->result = false;

  FREE(_data->sumMin.sums, _data->sumMin.num);
  FREE(_data->sumMin.mins, _data->sumMin.num);
  FREE(_data->sumMin.flags, _data->sumMin.num);
  FREE(_data->todoNodes.nodes, _data->todoNodes.num);
  FREE(_data->todoNodes.costs, _data->todoNodes.num);
  FREE(_data->todoSorted.nodes, _data->todoSorted.num);
  FREE(_data->todoSorted.costs, _data->todoSorted.num);
  FREE(_data->leafNodes, _data->numLeafNodes);
  FREE(_data->patches, _data->numPatches);
  FREE(_data->skipped, _data->numSkipped);
  FREE(_data->nodesToFix, _data->numNodesToFix);
  FREE(_data->cubSortTempStorage, _data->cubSortTempStorageSize);
  cudaProfilerStop();
}


void initData(GPUWorkingMemory *_data, CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  cudaError_t   lRes;
  uint32_t      lNumBlocksAll     = (_data->todoNodes.num + _blockSize - 1) / _blockSize;
  uint32_t      lNumBlocksPatches = (_data->numPatches + _blockSize - 1) / _blockSize;
  uint32_t      lNumBlocksInit    = (_data->numLeafNodes + _blockSize - 1) / _blockSize;
  void *        lTempStorage      = nullptr;
  int *         lNumSelected      = nullptr;
  size_t        lTempStorageSize  = 0;
  CUBLeafSelect lSelector(_GPUbvh->nodes);

  kResetTodoData<<<lNumBlocksAll, _blockSize>>>(_data->todoNodes.nodes, _data->todoNodes.num);
  kInitPatches<<<lNumBlocksPatches, _blockSize>>>(_data->patches, _GPUbvh->bvh, _data->numPatches);

  CUDA_RUN(cudaMemset(_data->sumMin.flags, 0, _data->sumMin.num * sizeof(uint32_t)));
  CUDA_RUN(cudaMemset(_data->skipped, 0, _data->numSkipped * sizeof(uint32_t)));

  ALLOCATE(&lNumSelected, 1, int);

  CUDA_RUN(cub::DeviceSelect::If(lTempStorage,
                                 lTempStorageSize,
                                 _data->todoNodes.nodes,
                                 _data->leafNodes,
                                 lNumSelected,
                                 _data->todoNodes.num,
                                 lSelector));

  ALLOCATE(&lTempStorage, lTempStorageSize, uint8_t);

  CUDA_RUN(cub::DeviceSelect::If(lTempStorage,
                                 lTempStorageSize,
                                 _data->todoNodes.nodes,
                                 _data->leafNodes,
                                 lNumSelected,
                                 _data->todoNodes.num,
                                 lSelector));

  kInitSumMin<<<lNumBlocksInit, _blockSize>>>(_data->leafNodes, _data->sumMin, _GPUbvh->nodes, _data->numLeafNodes);
  CUDA_RUN(cudaMemset(_data->sumMin.flags, 0, _data->sumMin.num * sizeof(uint32_t)));

error:
  cudaFree(lNumSelected);
  FREE(lTempStorage, lTempStorageSize);
}
