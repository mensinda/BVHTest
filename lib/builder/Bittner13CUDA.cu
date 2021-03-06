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

  __device__ __forceinline__ bool operator()(const uint32_t &a) const { return nodes[a].isLeafFlag != 0; }
};

struct CUBNodeSelect {
  BVHNode *nodes;

  CUB_RUNTIME_FUNCTION __forceinline__ CUBNodeSelect(BVHNode *_n) : nodes(_n) {}

  __device__ __forceinline__ bool operator()(const uint32_t &a) const { return nodes[a].isLeafFlag == 0; }
};

struct CUBNodeSlelect {
  float  k;
  float *c;

  CUB_RUNTIME_FUNCTION __forceinline__ CUBNodeSlelect(float *_c, float _k) : k(_k), c(_c) {}

  __device__ __forceinline__ bool operator()(const uint32_t &a) const { return c[a] >= k; }
};


extern "C" __global__ void kResetTodoData(uint32_t *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) { _nodes[i] = i; }
}

extern "C" __global__ void kInitPatches(BVHPatch *_patches, BVH *_bvh, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) { new (_patches + i) BVHPatch(_bvh); }
}

extern "C" __global__ void kGenerateFlags(float *_costs, uint8_t *_flagsOut, float *k, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) { _flagsOut[i] = _costs[i] >= *k ? 1 : 0; }
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

__device__ uint32_t findNode1(uint32_t _n, BVHPatch &_bvh) {
  float             lBestCost      = HUGE_VALF;
  uint32_t          lBestNodeIndex = 0;
  BVHNode const *   lNode          = _bvh.getOrig(_n);
  AABB              lNodeBBox      = lNode->bbox;
  float             lSArea         = lNode->surfaceArea;
  uint32_t          lSize          = 1;
  CUDAHelperStruct  lPQ[CUDA_QUEUE_SIZE];
  CUDAHelperStruct *lBegin = lPQ;

  lPQ[0] = {_bvh.root(), 0.0f, 0};
  while (lSize > 0) {
    CUDAHelperStruct lCurr = lPQ[0];

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    CUDA_pop_heap(lBegin, lBegin + lSize);
    lSize--;

    BVHNodePatch lCurrNode = _bvh.getSubset(lCurr.node);
    AABB         lBBox     = _bvh.getAABB(lCurr.node);

    float lNewInduced = -1 * lBBox.surfaceArea();

    lBBox.mergeWith(lNodeBBox);
    float lDirectCost = lBBox.surfaceArea();
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = lCurr.node;
    }

    lNewInduced += lTotalCost;
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


__device__ uint32_t findNode2(uint32_t _n, BVHPatch &_bvh) {
  float          lBestCost      = HUGE_VALF;
  uint32_t       lBestNodeIndex = 0;
  BVHNode const *lNode          = _bvh.getOrig(_n);
  AABB           lNodeBBox      = lNode->bbox;
  float          lSArea         = lNode->surfaceArea;
  float          lMin           = 0.0f;
  float          lMax           = HUGE_VALF;
  uint16_t       lStart         = threadIdx.x * CUDA_ALT_QUEUE_SIZE;
  uint16_t       lMinIndex      = 0;
  uint16_t       lMaxIndex      = 1;

  extern __shared__ uint32_t lPQ_NL[];
  float                      lPQ_CO[CUDA_ALT_QUEUE_SIZE];
  uint32_t                   lCurrNL;
  float                      lCurrCO;

  // Init
  for (uint32_t i = 0; i < CUDA_ALT_QUEUE_SIZE; ++i) { lPQ_CO[i] = HUGE_VALF; }

  lPQ_NL[lMinIndex + lStart] = _bvh.root();
  lPQ_CO[lMinIndex]          = 0.0f;
  while (lMin < HUGE_VALF) {
    lCurrNL           = lPQ_NL[lMinIndex + lStart];
    lCurrCO           = lPQ_CO[lMinIndex];
    lPQ_CO[lMinIndex] = HUGE_VALF;

    if ((lCurrCO + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    BVHNodePatch lCurrNode = _bvh.getSubset(lCurrNL);
    AABB         lBBox     = _bvh.getAABB(lCurrNL);

    float lNewInduced = -1 * lBBox.surfaceArea();

    lBBox.mergeWith(lNodeBBox);
    float lDirectCost = lBBox.surfaceArea();
    float lTotalCost  = lCurrCO + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = lCurrNL;
    }

    lNewInduced += lTotalCost;
    if ((lNewInduced + lSArea) < lBestCost && !lCurrNode.isLeaf()) {
      lPQ_NL[lMinIndex + lStart] = lCurrNode.left;
      lPQ_NL[lMaxIndex + lStart] = lCurrNode.right;
      lPQ_CO[lMinIndex]          = lNewInduced;
      lPQ_CO[lMaxIndex]          = lNewInduced;
    }

    lMin = HUGE_VALF;
    lMax = 0.0f;

    for (uint32_t i = 0; i < CUDA_ALT_QUEUE_SIZE; ++i) {
      if (lPQ_CO[i] < lMin) {
        lMin      = lPQ_CO[i];
        lMinIndex = i;
      }
      if (lPQ_CO[i] > lMax) {
        lMax      = lPQ_CO[i];
        lMaxIndex = i;
      }
    }
  }

  return lBestNodeIndex;
}


__device__ CUDA_RM_RES removeNode(uint32_t _node, BVHPatch &_bvh) {
  CUDA_RM_RES lFalse = {false, {0, 0}, {0, 0}, {0, 0}};
  if (_bvh.getOrig(_node)->isLeaf() || _node == _bvh.root()) { return lFalse; }

  BVHNodePatch *lNode         = _bvh.patchNode(_node, PINDEX_NODE);
  uint32_t      lSiblingIndex = _bvh.sibling(*lNode);
  uint32_t      lParentIndex  = lNode->parent;

  if (lParentIndex == _bvh.root()) { return lFalse; } // Can not remove node with this algorithm

  BVHNodePatch *lSibling          = _bvh.patchNode(lSiblingIndex, PINDEX_SIBLING);
  BVHNodePatch *lParent           = _bvh.patchNode(lParentIndex, PINDEX_PARENT);
  uint32_t      lGrandParentIndex = lParent->parent;

  BVHNodePatch *lGrandParent = _bvh.patchNode(lGrandParentIndex, PINDEX_GRAND_PARENT);

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
  _bvh.patchAABBFrom(lGrandParentIndex, 0);

  if (_bvh.getOrig(lNode->left)->surfaceArea > _bvh.getOrig(lNode->right)->surfaceArea) {
    return {true, {lNode->left, lNode->right}, {_node, lParentIndex}, {lGrandParentIndex, lSiblingIndex}};
  } else {
    return {true, {lNode->right, lNode->left}, {_node, lParentIndex}, {lGrandParentIndex, lSiblingIndex}};
  }
}


__device__ CUDA_INS_RES reinsert(uint32_t _node, uint32_t _unused, BVHPatch &_bvh, bool _update, bool _altFindNode) {
  uint32_t lBestIndex = _altFindNode ? findNode2(_node, _bvh) : findNode1(_node, _bvh);
  if (lBestIndex == _bvh.root()) { return {false, 0, 0}; }

  uint32_t      lBestPatchIndex = _bvh.patchIndex(lBestIndex); // Check if node is already patched
  BVHNodePatch *lBest           = nullptr;

  if (lBestPatchIndex == UINT32_MAX) {
    lBest = _bvh.patchNode(lBestIndex, _update ? PINDEX_1ST_BEST : PINDEX_2ND_BEST);
  } else if (lBestPatchIndex == PINDEX_GRAND_PARENT) {
    lBest = _bvh.getPatchedNode(PINDEX_GRAND_PARENT);
  } else {
    lBest = _bvh.movePatch(lBestPatchIndex, _update ? PINDEX_1ST_BEST : PINDEX_2ND_BEST);
  }

  BVHNodePatch *lNode           = _bvh.patchNode(_node, _update ? PINDEX_1ST_INSERT : PINDEX_2ND_INSERT);
  BVHNodePatch *lUnused         = _bvh.getPatchedNode(_update ? PINDEX_NODE : PINDEX_PARENT);
  uint32_t      lRootIndex      = lBest->parent;
  uint32_t      lRootPatchIndex = _bvh.patchIndex(lRootIndex);
  BVHNodePatch *lRoot           = nullptr;

  if (lRootPatchIndex == UINT32_MAX) {
    lRoot = _bvh.patchNode(lRootIndex, _update ? PINDEX_1ST_ROOT : PINDEX_2ND_ROOT);
  } else {
    lRoot = _bvh.movePatch(lRootPatchIndex, _update ? PINDEX_1ST_ROOT : PINDEX_2ND_ROOT);
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
  lUnused->left   = lBestIndex;
  lUnused->right  = _node;

  lBest->parent = _unused;
  lBest->isLeft = TRUE;
  lNode->parent = _unused;
  lNode->isLeft = FALSE;

  if (_update) { _bvh.patchAABBFrom(_unused, 1); }

  return {true, lBestIndex, lRootIndex};
}




/*  ______ _        _____               */
/*  |  ___(_)      |_   _|              */
/*  | |_   ___  __   | |_ __ ___  ___   */
/*  |  _| | \ \/ /   | | '__/ _ \/ _ \  */
/*  | |   | |>  <    | | | |  __/  __/  */
/*  \_|   |_/_/\_\   \_/_|  \___|\___|  */
/*                                      */
/*                                      */

extern "C" __global__ void kFixTree1(uint32_t *_leaf, uint32_t *_flags, BVHNode *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  AABB     lAABB;
  uint32_t lNode;
  uint32_t lLeft;
  uint32_t lRight;

  for (uint32_t i = index; i < _num; i += stride) {
    lNode = _leaf[i];
    lNode = _nodes[lNode].parent;

    while (true) {
      uint32_t lOldLock = atomicCAS(&_flags[lNode], 0, 1);

      // Check if this thread is first. If yes break
      if (lOldLock == 0) { break; }

      lLeft  = _nodes[lNode].left;
      lRight = _nodes[lNode].right;
      lAABB  = _nodes[lLeft].bbox;
      lAABB.mergeWith(_nodes[lRight].bbox);

      _nodes[lNode].bbox        = lAABB;
      _nodes[lNode].surfaceArea = lAABB.surfaceArea();

      // Check if root
      if (lNode == _nodes[lNode].parent) { break; }
      lNode = _nodes[lNode].parent;
    }
  }
}


extern "C" __global__ void kFixTree3_1(uint32_t *_toFix, uint32_t *_flags, BVHNode *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (uint32_t i = index; i < _num; i += stride) {
    uint32_t lNode = _toFix[i];
    if (lNode == UINT32_MAX) { continue; }

    while (true) {
      if (atomicAdd(&_flags[lNode], 1) != 0) { break; } // Stop when already locked (locked == 1)

      // Check if root
      if (lNode == _nodes[lNode].parent) { break; }
      lNode = _nodes[lNode].parent;
    }
  }
}

extern "C" __global__ void kFixTree3_2(uint32_t *_toFix, uint32_t *_flags, BVHNode *_nodes, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  AABB     lAABB;
  uint32_t lNode;
  uint32_t lLeft;
  uint32_t lRight;
  BVHNode  lNodeCPY;

  for (uint32_t i = index; i < _num; i += stride) {
    lNode = _toFix[i];
    if (lNode == UINT32_MAX) { continue; }

    lNodeCPY = _nodes[lNode];

    // Check if leaf node
    if (lNodeCPY.isLeafFlag != 0) {
      atomicSub(&_flags[lNode], 1); // Leaf nodes should only be locked once
      lNode    = lNodeCPY.parent;
      lNodeCPY = _nodes[lNode];
    }

    while (true) {
      if (atomicSub(&_flags[lNode], 1) != 1) { break; } // Stop when already locked (locked == 1)

      lLeft  = lNodeCPY.left;
      lRight = lNodeCPY.right;
      lAABB  = _nodes[lLeft].bbox;
      lAABB.mergeWith(_nodes[lRight].bbox);

      lNodeCPY.bbox        = lAABB;
      lNodeCPY.surfaceArea = lAABB.surfaceArea();

      _nodes[lNode] = lNodeCPY;

      // Check if root
      if (lNode == lNodeCPY.parent) { break; }
      lNode    = lNodeCPY.parent;
      lNodeCPY = _nodes[lNode];
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

extern "C" __global__ void kCalcCost(BVHNode *_nodes, uint32_t *_nID, float *_costOut, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) {
    uint    lID   = _nID[i];
    BVHNode lNode = _nodes[lID];
    float   lCost;

    if (lNode.isLeafFlag == 0) {
      float lSA      = lNode.surfaceArea;
      float lLeftSA  = _nodes[lNode.left].surfaceArea;
      float lRightSA = _nodes[lNode.right].surfaceArea;

      lCost = (lSA * lSA * lSA * 2.0f) / ((lLeftSA + lRightSA) * (lLeftSA < lRightSA ? lLeftSA : lRightSA));
    } else {
      lCost = 0;
    }

    _costOut[i] = lCost;
  }
}



extern "C" __global__ void kRemoveAndReinsert1(uint32_t *_todoList,
                                               BVHPatch *_patches,
                                               uint32_t *_toFix,
                                               bool      _offsetAccess,
                                               uint32_t  _chunk,
                                               uint32_t  _numChunks,
                                               uint32_t  _chunkSize,
                                               bool      _altFindNode) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (int32_t k = index; k < _chunkSize; k += stride) {
    uint32_t     lNodeIndex = _todoList[_offsetAccess ? k * _numChunks + _chunk : _chunk * _chunkSize + k];
    CUDA_RM_RES  lRmRes     = removeNode(lNodeIndex, _patches[k]);
    CUDA_INS_RES lR1, lR2;

    if (!lRmRes.res) { continue; }

    lR1 = reinsert(lRmRes.toInsert.n1, lRmRes.unused.n1, _patches[k], true, _altFindNode);
    lR2 = reinsert(lRmRes.toInsert.n2, lRmRes.unused.n2, _patches[k], false, _altFindNode);

    if (!lR1.res || !lR2.res) { continue; }

    _toFix[k * 3 + 0] = lRmRes.grandParentAndSibling.n1;
    _toFix[k * 3 + 1] = lRmRes.unused.n1;
    _toFix[k * 3 + 2] = lRmRes.unused.n2;
  }
}

extern "C" __global__ void kRemoveAndReinsert2(uint32_t *_todoList,
                                               BVHPatch *_patches,
                                               BVH *     _bvh,
                                               uint32_t *_toFix,
                                               bool      _offsetAccess,
                                               uint32_t  _chunk,
                                               uint32_t  _numChunks,
                                               uint32_t  _chunkSize,
                                               bool      _altFindNode) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  for (int32_t k = index; k < _chunkSize; k += stride) {
    BVHPatch     lPatch(_bvh);
    uint32_t     lNodeIndex = _todoList[_offsetAccess ? k * _numChunks + _chunk : _chunk * _chunkSize + k];
    CUDA_RM_RES  lRmRes     = removeNode(lNodeIndex, lPatch);
    CUDA_INS_RES lR1, lR2;

    if (!lRmRes.res) { continue; }

    lR1 = reinsert(lRmRes.toInsert.n1, lRmRes.unused.n1, lPatch, true, _altFindNode);
    lR2 = reinsert(lRmRes.toInsert.n2, lRmRes.unused.n2, lPatch, false, _altFindNode);

    if (!lR1.res || !lR2.res) { continue; }

    _toFix[k * 3 + 0] = lRmRes.grandParentAndSibling.n1;
    _toFix[k * 3 + 1] = lRmRes.unused.n1;
    _toFix[k * 3 + 2] = lRmRes.unused.n2;

    _patches[k] = lPatch;
  }
}


extern "C" __global__ void kCheckConflicts(
    BVHPatch *_patches, uint32_t *_flags, uint32_t *_skip, uint32_t *_toFix, uint32_t _num) {
  uint32_t index   = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride  = blockDim.x * gridDim.x;
  uint32_t lLockID = index + 1;

  for (uint32_t k = index; k < _num; k += stride) {
    uint32_t lPatch[NNode];
    for (uint32_t i = 0; i < NNode; ++i) { lPatch[i] = _patches[k].getPatchedNodeIndex(i); }

#define TRY_LOCK(N)                                                                                                    \
  if (lPatch[N] != UINT32_MAX) {                                                                                       \
    IF_NOT_LOCK(lPatch[N], lLockID) { goto FAILED; }                                                                   \
  }

    TRY_LOCK(0);
    TRY_LOCK(1);
    TRY_LOCK(2);
    TRY_LOCK(3);
    TRY_LOCK(4);
    TRY_LOCK(5);
    TRY_LOCK(6);
    TRY_LOCK(7);
    TRY_LOCK(8);
    TRY_LOCK(9);

#undef TRY_LOCK

    continue;

  FAILED:

    for (uint32_t i = 0; i < NNode; ++i) {
      if (lPatch[i] != UINT32_MAX) { RELEASE_LOCK_S(lPatch[i], lLockID); }
    }

    _skip[k] += 1;
    _toFix[k * 3 + 0] = UINT32_MAX;
    _toFix[k * 3 + 1] = UINT32_MAX;
    _toFix[k * 3 + 2] = UINT32_MAX;
    _patches[k].clear();
  }
}


extern "C" __global__ void kApplyPatches(BVHPatch *_patches, BVH *_bvh, uint32_t *_flags, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t k = index; k < _num; k += stride) {
    MINI_PATCH lPatch;
    _patches[k].genMiniPatch(lPatch);
    _patches[k].clear();

    for (uint32_t l = 0; l < NNode; ++l) {
      if (lPatch.vPatch[l] == UINT32_MAX) { continue; }

      RELEASE_LOCK(lPatch.vPatch[l]);
      lPatch.applyOne(l, _bvh);
    }
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
  kFixTree1<<<lNumBlocks, _blockSize>>>(_data->leafNodes, _data->flags, _GPUbvh->nodes, _data->numLeafNodes);

  cudaMemset(_data->flags, 0, _data->numFlags * sizeof(uint32_t));
}

void fixTree3(GPUWorkingMemory *_data, BVHTest::base::CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  uint32_t lNumBlocks = (_data->numNodesToFix + _blockSize - 1) / _blockSize;
  kFixTree3_1<<<lNumBlocks, _blockSize>>>(_data->nodesToFix, _data->flags, _GPUbvh->nodes, _data->numNodesToFix);
  kFixTree3_2<<<lNumBlocks, _blockSize>>>(_data->nodesToFix, _data->flags, _GPUbvh->nodes, _data->numNodesToFix);
}


void bn13_selectNodes(GPUWorkingMemory *_data, BVHTest::base::CUDAMemoryBVHPointer *_GPUbvh, AlgoCFG _cfg) {
  if (!_data || !_GPUbvh) { return; }

  cudaError_t lRes;
  uint32_t    lNBlkAll = (_data->numInnerNodes + _cfg.blockSize - 1) / _cfg.blockSize;

  int *lNumSelected = nullptr;

  ALLOCATE(&lNumSelected, 1, int);

  kCalcCost<<<lNBlkAll, _cfg.blockSize>>>(
      _GPUbvh->nodes, _data->todoNodes.nodes, _data->todoNodes.costs, _data->numInnerNodes);

  if (_cfg.sort) {
    if (!_cfg.altSort) {
      CUDA_RUN(cub::DeviceRadixSort::SortPairsDescending(_data->cubSortTempStorage,
                                                         _data->cubSortTempStorageSize,
                                                         _data->todoNodes.costs,
                                                         _data->todoSorted.costs,
                                                         _data->todoNodes.nodes,
                                                         _data->todoSorted.nodes,
                                                         _data->numInnerNodes));
    } else {
      CUDA_RUN(cub::DeviceRadixSort::SortKeysDescending(_data->cubSortTempStorage,
                                                        _data->cubSortTempStorageSize,
                                                        _data->todoNodes.costs,
                                                        _data->todoSorted.costs,
                                                        _data->numInnerNodes));

      kGenerateFlags<<<lNBlkAll, _cfg.blockSize>>>(_data->todoNodes.costs,
                                                   _data->deviceSelectFlags,
                                                   _data->todoSorted.costs + _cfg.numChunks * _cfg.chunkSize,
                                                   _data->numInnerNodes);

      CUDA_RUN(cub::DeviceSelect::Flagged(_data->cubSortTempStorage,
                                          _data->cubSortTempStorageSize,
                                          _data->todoNodes.nodes,
                                          _data->deviceSelectFlags,
                                          _data->todoSorted.nodes,
                                          lNumSelected,
                                          _data->numInnerNodes));
    }

  } else {
#if 0
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);

    float lKCost = topKThElement(_data->todoNodes.costs, _data->sumMin.num, _cfg.numChunks * _cfg.chunkSize);

    CUBNodeSlelect lSelector(_data->todoNodes.costs, lKCost);

    CUDA_RUN(cub::DeviceSelect::If(_data->cubSortTempStorage,
                                   _data->cubSortTempStorageSize,
                                   _data->todoNodes.nodes,
                                   _data->todoSorted.nodes,
                                   lNumSelected,
                                   _data->numInnerNodes,
                                   lSelector));
#else
    cerr << "ERROR: topKThElement is currently not supported (set option to sort)" << endl;
#endif
  }

error:
  cudaFree(lNumSelected);
  return;
}


void bn13_rmAndReinsChunk(GPUWorkingMemory *_data, CUDAMemoryBVHPointer *_GPUbvh, AlgoCFG _cfg, uint32_t _chunk) {
  if (!_data || !_GPUbvh) { return; }

  uint32_t lNBlkChunk    = (_cfg.chunkSize + _cfg.blockSize - 1) / _cfg.blockSize;
  uint32_t lSharedMemory = sizeof(uint32_t) * _cfg.blockSize * CUDA_ALT_QUEUE_SIZE;

  if (_cfg.localPatchCPY) {
    kRemoveAndReinsert2<<<lNBlkChunk, _cfg.blockSize, lSharedMemory>>>(_data->todoSorted.nodes,
                                                                       _data->patches,
                                                                       _GPUbvh->bvh,
                                                                       _data->nodesToFix,
                                                                       _cfg.offsetAccess,
                                                                       _chunk,
                                                                       _cfg.numChunks,
                                                                       _cfg.chunkSize,
                                                                       _cfg.altFindNode);
  } else {
    kRemoveAndReinsert1<<<lNBlkChunk, _cfg.blockSize, lSharedMemory>>>(_data->todoSorted.nodes,
                                                                       _data->patches,
                                                                       _data->nodesToFix,
                                                                       _cfg.offsetAccess,
                                                                       _chunk,
                                                                       _cfg.numChunks,
                                                                       _cfg.chunkSize,
                                                                       _cfg.altFindNode);
  }

  kCheckConflicts<<<lNBlkChunk, _cfg.blockSize>>>(
      _data->patches, _data->flags, _data->skipped, _data->nodesToFix, _cfg.chunkSize);

  kApplyPatches<<<lNBlkChunk, _cfg.blockSize>>>(_data->patches, _GPUbvh->bvh, _data->flags, _cfg.chunkSize);

  if (_cfg.altFixTree) {
    fixTree3(_data, _GPUbvh, _cfg.blockSize);
  } else {
    fixTree1(_data, _GPUbvh, _cfg.blockSize);
  }
}

void bn13_doAlgorithmStep(GPUWorkingMemory *_data, CUDAMemoryBVHPointer *_GPUbvh, AlgoCFG _cfg) {
  bn13_selectNodes(_data, _GPUbvh, _cfg);

  for (uint32_t i = 0; i < _cfg.numChunks; ++i) { bn13_rmAndReinsChunk(_data, _GPUbvh, _cfg, i); }
}



float CUDAcalcSAH(CUDAMemoryBVHPointer *_GPUbvh) {
  cudaError_t lRes;
  BVHNode     lResNode;
  void *      lTemp       = nullptr;
  size_t      lTempS      = 0;
  BVHNode *   lResNodeGPU = nullptr;
  float       lSAH        = 0.0f;
  float       lSumAll     = 0.0f; // Sum of all surfaceAreas

  ALLOCATE(&lResNodeGPU, 1, BVHNode);
  CUDA_RUN(cub::DeviceReduce::Sum(lTemp, lTempS, _GPUbvh->nodes, lResNodeGPU, _GPUbvh->numNodes));
  ALLOCATE(&lTemp, lTempS, uint8_t);
  CUDA_RUN(cub::DeviceReduce::Sum(lTemp, lTempS, _GPUbvh->nodes, lResNodeGPU, _GPUbvh->numNodes));
  CUDA_RUN(cudaMemcpy(&lResNode, lResNodeGPU, sizeof(BVHNode), cudaMemcpyDeviceToHost));

  lSumAll = lResNode.surfaceArea;

  // Get the root node
  CUDA_RUN(cudaMemcpy(&lResNode, _GPUbvh->nodes + 0, sizeof(BVHNode), cudaMemcpyDeviceToHost));

  lSumAll -= lResNode.surfaceArea; // Subtract the SA of the root node, that was added in the Sum above
  lSAH = lSumAll / lResNode.surfaceArea;

error:
  FREE2(lResNodeGPU);
  FREE2(lTemp);
  return lSAH;
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
  size_t      lCubTempStorage3 = 0;
  size_t      lCubTempStorage4 = 0;
  size_t      lCubTempStorageM = 0;
  uint32_t *  lTemp            = nullptr;

  CUBLeafSelect lSelector(nullptr);

  assert(_bvh->numNodes > _numFaces);
  lMem.numLeafNodes  = _numFaces;
  lMem.numInnerNodes = _bvh->numNodes - _numFaces;
  lMem.numPatches    = _batchSize;
  lMem.numSkipped    = _batchSize;
  lMem.numNodesToFix = _batchSize * 3;
  lMem.numFlags      = _bvh->numNodes;

  ALLOCATE(&lMem.flags, lMem.numFlags, uint32_t);
  ALLOCATE(&lMem.todoNodes.nodes, lMem.numInnerNodes, uint32_t);
  ALLOCATE(&lMem.todoNodes.costs, lMem.numInnerNodes, float);
  ALLOCATE(&lMem.todoSorted.nodes, lMem.numInnerNodes, uint32_t);
  ALLOCATE(&lMem.todoSorted.costs, lMem.numInnerNodes, float);
  ALLOCATE(&lMem.leafNodes, lMem.numLeafNodes, uint32_t);
  ALLOCATE(&lMem.deviceSelectFlags, lMem.numLeafNodes, uint8_t);
  ALLOCATE(&lMem.patches, lMem.numPatches, BVHPatch);
  ALLOCATE(&lMem.skipped, lMem.numSkipped, uint32_t);
  ALLOCATE(&lMem.nodesToFix, lMem.numNodesToFix, uint32_t);



  // This only calculates the memory requirements
  CUDA_RUN(cub::DeviceRadixSort::SortPairsDescending(lMem.cubSortTempStorage,
                                                     lCubTempStorage1,
                                                     lMem.todoNodes.costs,
                                                     lMem.todoSorted.costs,
                                                     lMem.todoNodes.nodes,
                                                     lMem.todoSorted.nodes,
                                                     lMem.numInnerNodes));

  CUDA_RUN(cub::DeviceRadixSort::SortKeysDescending(
      lMem.cubSortTempStorage, lCubTempStorage4, lMem.todoNodes.costs, lMem.todoSorted.costs, lMem.numInnerNodes));

  CUDA_RUN(cub::DeviceSelect::If(lMem.cubSortTempStorage,
                                 lCubTempStorage3,
                                 lMem.todoNodes.nodes,
                                 lMem.leafNodes,
                                 lTemp,
                                 (lMem.numLeafNodes + lMem.numInnerNodes),
                                 lSelector));

  CUDA_RUN(cub::DeviceReduce::Sum(lMem.cubSortTempStorage, lCubTempStorage2, lMem.skipped, lTemp, lMem.numSkipped));

  lCubTempStorageM            = lCubTempStorage1 > lCubTempStorage2 ? lCubTempStorage1 : lCubTempStorage2;
  lCubTempStorageM            = lCubTempStorageM > lCubTempStorage3 ? lCubTempStorageM : lCubTempStorage3;
  lCubTempStorageM            = lCubTempStorageM > lCubTempStorage4 ? lCubTempStorageM : lCubTempStorage4;
  lMem.cubSortTempStorageSize = lCubTempStorageM;

  ALLOCATE(&lMem.cubSortTempStorage, lMem.cubSortTempStorageSize, uint8_t);

  return lMem;

error:
  lMem.result = false;

  FREE(lMem.flags, lMem.numFlags);
  FREE(lMem.todoNodes.nodes, lMem.numInnerNodes);
  FREE(lMem.todoNodes.costs, lMem.numInnerNodes);
  FREE(lMem.todoSorted.nodes, lMem.numInnerNodes);
  FREE(lMem.todoSorted.costs, lMem.numInnerNodes);
  FREE(lMem.leafNodes, lMem.numLeafNodes);
  FREE(lMem.deviceSelectFlags, lMem.numLeafNodes);
  FREE(lMem.patches, lMem.numPatches);
  FREE(lMem.skipped, lMem.numSkipped);
  FREE(lMem.nodesToFix, lMem.numNodesToFix);
  FREE(lMem.cubSortTempStorage, lMem.cubSortTempStorageSize);

  return lMem;
}

void freeMemory(GPUWorkingMemory *_data) {
  _data->result = false;

  FREE(_data->flags, _data->numFlags);
  FREE(_data->todoNodes.nodes, _data->numInnerNodes);
  FREE(_data->todoNodes.costs, _data->numInnerNodes);
  FREE(_data->todoSorted.nodes, _data->numInnerNodes);
  FREE(_data->todoSorted.costs, _data->numInnerNodes);
  FREE(_data->leafNodes, _data->numLeafNodes);
  FREE(_data->deviceSelectFlags, _data->numLeafNodes);
  FREE(_data->patches, _data->numPatches);
  FREE(_data->skipped, _data->numSkipped);
  FREE(_data->nodesToFix, _data->numNodesToFix);
  FREE(_data->cubSortTempStorage, _data->cubSortTempStorageSize);
  cudaProfilerStop();
}


void initData(GPUWorkingMemory *_data, CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  cudaError_t   lRes;
  uint32_t      lNumBlocksAll     = (_data->numInnerNodes + _blockSize - 1) / _blockSize;
  uint32_t      lNumBlocksPatches = (_data->numPatches + _blockSize - 1) / _blockSize;
  int *         lNumSelected      = nullptr;
  int           lNS1              = 0;
  int           lNS2              = 0;
  CUBLeafSelect lSelector1(_GPUbvh->nodes);
  CUBNodeSelect lSelector2(_GPUbvh->nodes);

  uint32_t *lTempTODOData = nullptr;
  ALLOCATE(&lTempTODOData, (_data->numLeafNodes + _data->numInnerNodes), uint32_t); // Allocate Inner + Leaf Nodes

  kResetTodoData<<<lNumBlocksAll, _blockSize>>>(lTempTODOData, (_data->numLeafNodes + _data->numInnerNodes));
  kInitPatches<<<lNumBlocksPatches, _blockSize>>>(_data->patches, _GPUbvh->bvh, _data->numPatches);

  CUDA_RUN(cudaMemset(_data->flags, 0, _data->numFlags * sizeof(uint32_t)));
  CUDA_RUN(cudaMemset(_data->skipped, 0, _data->numSkipped * sizeof(uint32_t)));

  ALLOCATE(&lNumSelected, 1, int);

  CUDA_RUN(cub::DeviceSelect::If(_data->cubSortTempStorage,
                                 _data->cubSortTempStorageSize,
                                 lTempTODOData,
                                 _data->leafNodes,
                                 lNumSelected,
                                 (_data->numLeafNodes + _data->numInnerNodes),
                                 lSelector1));

  CUDA_RUN(cudaMemcpy(&lNS1, lNumSelected, sizeof(int), cudaMemcpyDeviceToHost));

  CUDA_RUN(cub::DeviceSelect::If(_data->cubSortTempStorage,
                                 _data->cubSortTempStorageSize,
                                 lTempTODOData,
                                 _data->todoNodes.nodes,
                                 lNumSelected,
                                 (_data->numLeafNodes + _data->numInnerNodes),
                                 lSelector2));

  CUDA_RUN(cudaMemcpy(&lNS2, lNumSelected, sizeof(int), cudaMemcpyDeviceToHost));

  assert(lNS1 == _data->numLeafNodes);
  assert(lNS2 == _data->numInnerNodes);
error:
  cudaFree(lNumSelected);
  cudaFree(lTempTODOData);
}
