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
#include <cub/cub.cuh>
#include <iostream>

using namespace glm;
using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::builder;
using namespace BVHTest::cuda;

#define CUDA_RUN(call)                                                                                                 \
  lRes = call;                                                                                                         \
  if (lRes != cudaSuccess) { goto error; }

#define ALLOCATE(ptr, num, type) CUDA_RUN(cudaMalloc(ptr, num * sizeof(type)));
#define FREE(ptr, num)                                                                                                 \
  cudaFree(ptr);                                                                                                       \
  ptr = nullptr;                                                                                                       \
  num = 0;

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

__global__ void kResetLocks(uint32_t *_locks, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) { _locks[i] = 0; }
}

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


extern "C" GPUWorkingMemory allocateMemory(CUDAMemoryBVHPointer *_bvh, uint32_t _batchSize, uint32_t _numFaces) {
  GPUWorkingMemory lMem;

  lMem.result = true;
  cudaError_t lRes;

  lMem.sumMin.num    = _bvh->numNodes;
  lMem.todoNodes.num = _bvh->numNodes;
  lMem.numLeafNodes  = _numFaces;
  lMem.numPatches    = _batchSize;

  ALLOCATE(&lMem.sumMin.sums, lMem.sumMin.num, float);
  ALLOCATE(&lMem.sumMin.mins, lMem.sumMin.num, float);
  ALLOCATE(&lMem.sumMin.flags, lMem.sumMin.num, uint32_t);
  ALLOCATE(&lMem.todoNodes.nodes, lMem.todoNodes.num, uint32_t);
  ALLOCATE(&lMem.todoNodes.costs, lMem.todoNodes.num, float);
  ALLOCATE(&lMem.leafNodes, lMem.numLeafNodes, uint32_t);
  ALLOCATE(&lMem.patches, lMem.numPatches, PATCH);

  return lMem;

error:
  lMem.result = false;

  FREE(lMem.sumMin.sums, lMem.sumMin.num);
  FREE(lMem.sumMin.mins, lMem.sumMin.num);
  FREE(lMem.sumMin.flags, lMem.sumMin.num);
  FREE(lMem.todoNodes.nodes, lMem.todoNodes.num);
  FREE(lMem.todoNodes.costs, lMem.todoNodes.num);
  FREE(lMem.leafNodes, lMem.numLeafNodes);
  FREE(lMem.patches, lMem.numPatches);

  return lMem;
}

extern "C" void freeMemory(GPUWorkingMemory *_data) {
  _data->result = false;

  FREE(_data->sumMin.sums, _data->sumMin.num);
  FREE(_data->sumMin.mins, _data->sumMin.num);
  FREE(_data->sumMin.flags, _data->sumMin.num);
  FREE(_data->todoNodes.nodes, _data->todoNodes.num);
  FREE(_data->todoNodes.costs, _data->todoNodes.num);
  FREE(_data->leafNodes, _data->numLeafNodes);
  FREE(_data->patches, _data->numPatches);
}


extern "C" void initData(GPUWorkingMemory *_data, CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  uint32_t lNumBlocks = (_data->todoNodes.num + _blockSize - 1) / _blockSize;
  kResetTodoData<<<lNumBlocks, _blockSize>>>(_data->todoNodes.nodes, _data->todoNodes.num);

  resetLocks(_data, _blockSize);

  cudaError_t   lRes;
  CUBLeafSelect lSelector(_GPUbvh->nodes);
  void *        lTempStorage     = nullptr;
  int *         lNumSelected     = nullptr;
  size_t        lTempStorageSize = 0;

  ALLOCATE(&lNumSelected, 1, int);

  cub::DeviceSelect::If(lTempStorage,
                        lTempStorageSize,
                        _data->todoNodes.nodes,
                        _data->leafNodes,
                        lNumSelected,
                        _data->todoNodes.num,
                        lSelector);

  ALLOCATE(&lTempStorage, lTempStorageSize, uint8_t);

  cub::DeviceSelect::If(lTempStorage,
                        lTempStorageSize,
                        _data->todoNodes.nodes,
                        _data->leafNodes,
                        lNumSelected,
                        _data->todoNodes.num,
                        lSelector);

error:

  cudaFree(lNumSelected);
  FREE(lTempStorage, lTempStorageSize);
}


extern "C" void fixTree(GPUWorkingMemory *_data, base::CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize) {
  if (!_data || !_GPUbvh) { return; }

  uint32_t lNumBlocks = (_data->sumMin.num + _blockSize - 1) / _blockSize;
  kFixTree<<<lNumBlocks, _blockSize>>>(_data->leafNodes,
                                       _data->sumMin.sums,
                                       _data->sumMin.mins,
                                       _GPUbvh->nodes,
                                       _data->sumMin.flags,
                                       _data->numLeafNodes);

  resetLocks(_data, _blockSize);
}

extern "C" void resetLocks(GPUWorkingMemory *_data, uint32_t _blockSize) {
  if (!_data) { return; }

  uint32_t lNumBlocks = (_data->sumMin.num + _blockSize - 1) / _blockSize;
  kResetLocks<<<lNumBlocks, _blockSize>>>(_data->sumMin.flags, _data->sumMin.num);
}
