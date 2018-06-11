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

#include "cuda/cudaFN.hpp"
#include "Bittner13CUDA.hpp"
#include "CUDAHeap.hpp"
#include <cmath>
#include <cub/cub.cuh>
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

// #define SPINN_LOCK(N)
#define IF_NOT_LOCK(N) if (atomicCAS(_flags + N, 0, 1) != 0)
#define RELEASE_LOCK(N) atomicExch(_flags + N, 0u);

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
  uint32_t grandParent;
};

__device__ CUDANodeLevel findNodeForReinsertion(uint32_t _n, PATCH &_bvh) {
  float             lBestCost      = HUGE_VALF;
  CUDANodeLevel     lBestNodeIndex = {0, 0};
  BVHNode const *   lNode          = _bvh[_n];
  AABB const &      lNodeBBox      = lNode->bbox;
  float             lSArea         = lNode->surfaceArea;
  uint32_t          lSize          = 1;
  CUDAHelperStruct  lPQ[CUDA_QUEUE_SIZE];
  CUDAHelperStruct *lBegin = lPQ;

  lPQ[0] = {_bvh.root(), 0.0f, 0};
  while (lSize > 0) {
    CUDAHelperStruct lCurr     = lPQ[0];
    BVHNode *        lCurrNode = _bvh[lCurr.node];
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
      if (!lCurrNode->isLeaf()) {
        assert(lSize + 2 < CUDA_QUEUE_SIZE);
        lPQ[lSize + 0] = {lCurrNode->left, lNewInduced, lCurr.level + 1};
        lPQ[lSize + 1] = {lCurrNode->right, lNewInduced, lCurr.level + 1};
        CUDA_push_heap(lBegin, lBegin + lSize + 1);
        CUDA_push_heap(lBegin, lBegin + lSize + 2);
        lSize += 2;
      }
    }
  }

  return lBestNodeIndex;
}


__device__ CUDA_RM_RES removeNode(uint32_t _node, PATCH &_bvh, uint32_t *_flags) {
  CUDA_RM_RES lFalse = {false, {0, 0}, {0, 0}, 0};
  if (_node == _bvh.root()) { return lFalse; }

  IF_NOT_LOCK(_node) { return lFalse; }

  BVHNode *lNode         = _bvh.patchNode(_node);
  uint32_t lSiblingIndex = _bvh.sibling(*lNode);
  uint32_t lParentIndex  = lNode->parent;

  IF_NOT_LOCK(lSiblingIndex) {
    RELEASE_LOCK(_node);
    return lFalse;
  }
  BVHNode *lSibling = _bvh.patchNode(lSiblingIndex);

  IF_NOT_LOCK(lParentIndex) {
    RELEASE_LOCK(_node);
    RELEASE_LOCK(lSiblingIndex);
    return lFalse;
  }
  BVHNode *lParent           = _bvh.patchNode(lParentIndex);
  uint32_t lGrandParentIndex = lParent->parent;

  IF_NOT_LOCK(lGrandParentIndex) {
    RELEASE_LOCK(_node);
    RELEASE_LOCK(lSiblingIndex);
    RELEASE_LOCK(lParentIndex);
    return lFalse;
  }
  BVHNode *lGrandParent = _bvh.patchNode(lGrandParentIndex);

  IF_NOT_LOCK(lNode->left) {
    RELEASE_LOCK(_node);
    RELEASE_LOCK(lSiblingIndex);
    RELEASE_LOCK(lParentIndex);
    RELEASE_LOCK(lGrandParentIndex);
    return lFalse;
  }

  IF_NOT_LOCK(lNode->right) {
    RELEASE_LOCK(_node);
    RELEASE_LOCK(lSiblingIndex);
    RELEASE_LOCK(lParentIndex);
    RELEASE_LOCK(lGrandParentIndex);
    RELEASE_LOCK(lNode->left);
    return lFalse;
  }

  BVHNode *lLeft  = _bvh.patchNode(lNode->left);
  BVHNode *lRight = _bvh.patchNode(lNode->right);

  // FREE LIST:   lNode, lParent
  // INSERT LIST: lLeft, lRight

  float lLeftSA  = lLeft->surfaceArea;
  float lRightSA = lRight->surfaceArea;


  if (lParentIndex == _bvh.root()) { return lFalse; } // Can not remove node with this algorithm

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

  if (lLeftSA > lRightSA) {
    return {true, {lNode->left, lNode->right}, {_node, lParentIndex}, lGrandParentIndex};
  } else {
    return {true, {lNode->right, lNode->left}, {_node, lParentIndex}, lGrandParentIndex};
  }
}


__device__ bool reinsert(uint32_t _node, uint32_t _unused, PATCH &_bvh, bool _update, uint32_t *_flags) {
  CUDANodeLevel lRes = findNodeForReinsertion(_node, _bvh);
  if (lRes.node == _bvh.root()) { return false; }

  uint32_t lBestPatchIndex = _bvh.patchIndex(lRes.node); // Check if node is already patched
  BVHNode *lBest           = nullptr;

  if (lBestPatchIndex == UINT32_MAX) {
    // Node is not patched ==> try to lock it
    IF_NOT_LOCK(lRes.node) { return false; }
    lBest = _bvh.patchNode(lRes.node);
  } else {
    // Node is already owned by this thread ==> no need to lock it
    lBest = _bvh.getPatchedNode(lBestPatchIndex);
  }

  BVHNode *lNode           = _bvh[_node];
  BVHNode *lUnused         = _bvh[_unused];
  uint32_t lRootIndex      = lBest->parent;
  uint32_t lRootPatchIndex = _bvh.patchIndex(lRootIndex);
  BVHNode *lRoot           = nullptr;

  if (lRootPatchIndex == UINT32_MAX) {
    IF_NOT_LOCK(lRootIndex) {
      RELEASE_LOCK(lRes.node);
      return false;
    }
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

  if (_update) {
    _bvh.nodeUpdated(lRes.node, lRes.level);
    _bvh.patchAABBFrom(_unused);
  }

  return true;
}




/*  ___  ___      _         _                        _       */
/*  |  \/  |     (_)       | |                      | |      */
/*  | .  . | __ _ _ _ __   | | _____ _ __ _ __   ___| |___   */
/*  | |\/| |/ _` | | '_ \  | |/ / _ \ '__| '_ \ / _ \ / __|  */
/*  | |  | | (_| | | | | | |   <  __/ |  | | | |  __/ \__ \  */
/*  \_|  |_/\__,_|_|_| |_| |_|\_\___|_|  |_| |_|\___|_|___/  */
/*                                                           */
/*                                                           */



__global__ void kFixTree(uint32_t *_leaf, float *_sum, float *_min, BVHNode *_node, uint32_t *_flag, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  AABB     lAABB;
  uint32_t lNode;
  uint32_t lLeft;
  uint32_t lRight;
  float    lSArea;

  for (uint32_t i = index; i < _num; i += stride) {
    lNode       = _leaf[i];
    _sum[lNode] = _node[lNode].surfaceArea;
    _min[lNode] = _node[lNode].surfaceArea;
    lNode       = _node[lNode].parent;

    while (true) {
      uint32_t lOldLock = atomicCAS(&_flag[lNode], 0, 1);

      // Check if this thread is first. If yes break
      if (lOldLock == 0) { break; }

      lLeft  = _node[lNode].left;
      lRight = _node[lNode].right;
      lAABB  = _node[lLeft].bbox;
      lAABB.mergeWith(_node[lRight].bbox);
      lSArea = lAABB.surfaceArea();

      _node[lNode].bbox        = lAABB;
      _node[lNode].surfaceArea = lSArea;
      _node[lNode].numChildren = _node[lLeft].numChildren + _node[lRight].numChildren + 2;
      _sum[lNode]              = _sum[lLeft] + _sum[lRight] + lSArea;
      _min[lNode]              = _min[lLeft] < _min[lRight] ? _min[lLeft] : _min[lRight];

      // Check if root
      if (lNode == _node[lNode].parent) { break; }
      lNode = _node[lNode].parent;
    }
  }
}


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



__global__ void kRemoveAndReinsert(uint32_t *_todoList,
                                   PATCH *   _patches,
                                   uint32_t *_flags,
                                   uint32_t *_skip,
                                   bool      _offsetAccess,
                                   uint32_t  _chunk,
                                   uint32_t  _numChunks,
                                   uint32_t  _chunkSize) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t k = index; k < _chunkSize; k += stride) {
    uint32_t    lNodeIndex = _todoList[_offsetAccess ? k * _numChunks + _chunk : _chunk * _chunkSize + k];
    CUDA_RM_RES lRmRes     = removeNode(lNodeIndex, _patches[k], _flags);

    if (!lRmRes.res) {
      _skip[k] += 1;
      _patches[k].clear();
      continue;
    }

    bool lR1 = reinsert(lRmRes.toInsert.n1, lRmRes.unused.n1, _patches[k], true, _flags);
    bool lR2 = reinsert(lRmRes.toInsert.n2, lRmRes.unused.n2, _patches[k], false, _flags);
    if (!lR1 || !lR2) {
      _skip[k] += 1;
      _patches[k].clear();

      // Unlock Nodes
      RELEASE_LOCK(lNodeIndex);
      RELEASE_LOCK(lRmRes.toInsert.n1);
      RELEASE_LOCK(lRmRes.toInsert.n2);
      RELEASE_LOCK(lRmRes.unused.n1);
      RELEASE_LOCK(lRmRes.unused.n2);
      RELEASE_LOCK(lRmRes.grandParent);
      continue;
    }
  }
}


__global__ void kApplyPatches(PATCH *_patches, uint32_t *_flags, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t k = index; k < _num; k += stride) {
    for (uint32_t l = 0; l < 10; ++l) {
      if (l >= _patches[k].size()) { break; }
      RELEASE_LOCK(_patches[k].getPatchedNodeIndex(l));
    }

    _patches[k].apply();
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


void fixTree(GPUWorkingMemory *_data, base::CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  uint32_t lNumBlocks = (_data->numLeafNodes + _blockSize - 1) / _blockSize;
  kFixTree<<<lNumBlocks, _blockSize>>>(_data->leafNodes,
                                       _data->sumMin.sums,
                                       _data->sumMin.mins,
                                       _GPUbvh->nodes,
                                       _data->sumMin.flags,
                                       _data->numLeafNodes);

  cudaMemset(_data->sumMin.flags, 0, _data->sumMin.num * sizeof(uint32_t));
}



void doAlgorithmStep(GPUWorkingMemory *    _data,
                     CUDAMemoryBVHPointer *_GPUbvh,
                     uint32_t              _numChunks,
                     uint32_t              _chunkSize,
                     uint32_t              _blockSize,
                     bool                  _offsetAccess) {
  if (!_data || !_GPUbvh) { return; }

  cudaError_t lRes;
  uint32_t    lNumBlocksAll   = (_data->sumMin.num + _blockSize - 1) / _blockSize;
  uint32_t    lNumBlocksChunk = (_chunkSize + _blockSize - 1) / _blockSize;

  /*   _____       _            _       _         _____           _     */
  /*  /  __ \     | |          | |     | |       /  __ \         | |    */
  /*  | /  \/ __ _| | ___ _   _| | __ _| |_ ___  | /  \/ ___  ___| |_   */
  /*  | |    / _` | |/ __| | | | |/ _` | __/ _ \ | |    / _ \/ __| __|  */
  /*  | \__/\ (_| | | (__| |_| | | (_| | ||  __/ | \__/\ (_) \__ \ |_   */
  /*   \____/\__,_|_|\___|\__,_|_|\__,_|\__\___|  \____/\___/|___/\__|  */
  /*                                                                    */
  /*                                                                    */

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
    kRemoveAndReinsert<<<lNumBlocksChunk, _blockSize>>>(_data->todoSorted.nodes,
                                                        _data->patches,
                                                        _data->sumMin.flags,
                                                        _data->skipped,
                                                        _offsetAccess,
                                                        i,
                                                        _numChunks,
                                                        _chunkSize);

    fixTree(_data, _GPUbvh, _blockSize);
  }

error:
  return;
}




/*  ___  ___                                                                                              _     */
/*  |  \/  |                                                                                             | |    */
/*  | .  . | ___ _ __ ___   ___  _ __ _   _   _ __ ___   __ _ _ __   __ _  __ _  ___ _ __ ___   ___ _ __ | |_   */
/*  | |\/| |/ _ \ '_ ` _ \ / _ \| '__| | | | | '_ ` _ \ / _` | '_ \ / _` |/ _` |/ _ \ '_ ` _ \ / _ \ '_ \| __|  */
/*  | |  | |  __/ | | | | | (_) | |  | |_| | | | | | | | (_| | | | | (_| | (_| |  __/ | | | | |  __/ | | | |_   */
/*  \_|  |_/\___|_| |_| |_|\___/|_|   \__, | |_| |_| |_|\__,_|_| |_|\__,_|\__, |\___|_| |_| |_|\___|_| |_|\__|  */
/*                                     __/ |                               __/ |                                */
/*                                    |___/                               |___/                                 */


GPUWorkingMemory allocateMemory(CUDAMemoryBVHPointer *_bvh, uint32_t _batchSize, uint32_t _numFaces) {
  GPUWorkingMemory lMem;

  lMem.result = true;
  cudaError_t lRes;

  lMem.sumMin.num     = _bvh->numNodes;
  lMem.todoNodes.num  = _bvh->numNodes;
  lMem.todoSorted.num = _bvh->numNodes;
  lMem.numLeafNodes   = _numFaces;
  lMem.numPatches     = _batchSize;
  lMem.numSkipped     = _batchSize;

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

  // This only calculates the memory requirements
  CUDA_RUN(cub::DeviceRadixSort::SortPairsDescending(lMem.cubSortTempStorage,
                                                     lMem.cubSortTempStorageSize,
                                                     lMem.todoNodes.costs,
                                                     lMem.todoSorted.costs,
                                                     lMem.todoNodes.nodes,
                                                     lMem.todoSorted.nodes,
                                                     lMem.todoNodes.num));

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
  FREE(_data->cubSortTempStorage, _data->cubSortTempStorageSize);
}


void initData(GPUWorkingMemory *_data, CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  cudaError_t   lRes;
  uint32_t      lNumBlocksAll     = (_data->todoNodes.num + _blockSize - 1) / _blockSize;
  uint32_t      lNumBlocksPatches = (_data->numPatches + _blockSize - 1) / _blockSize;
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

error:
  cudaFree(lNumSelected);
  FREE(lTempStorage, lTempStorageSize);
}
