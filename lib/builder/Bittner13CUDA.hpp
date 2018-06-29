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

#include "base/BVH.hpp"
#include "base/BVHPatch.hpp"
#include <cstdint>

typedef BVHTest::base::BVHPatch  PATCH;
typedef BVHTest::base::MiniPatch MINI_PATCH;
const size_t                     CUDA_QUEUE_SIZE     = 512;
const size_t                     CUDA_ALT_QUEUE_SIZE = 16;

struct SumMinCUDA {
  float *   sums  = nullptr;
  float *   mins  = nullptr;
  uint32_t *flags = nullptr;
  uint32_t  num   = 0;
};

struct TodoStruct {
  uint32_t *nodes = nullptr;
  float *   costs = nullptr;
  uint32_t  num   = 0;
};

struct GPUWorkingMemory {
  bool       result;
  SumMinCUDA sumMin;
  TodoStruct todoNodes;
  TodoStruct todoSorted;

  uint32_t *leafNodes              = nullptr;
  PATCH *   patches                = nullptr;
  uint32_t *skipped                = nullptr;
  uint32_t *nodesToFix             = nullptr;
  void *    cubSortTempStorage     = nullptr;
  uint32_t  numLeafNodes           = 0;
  uint32_t  numPatches             = 0;
  uint32_t  numSkipped             = 0;
  uint32_t  numNodesToFix          = 0;
  size_t    cubSortTempStorageSize = 0;
};

struct AlgoCFG {
  uint32_t blockSize     = 32;
  bool     offsetAccess  = true;
  bool     altFindNode   = true;
  bool     altFixTree    = true;
  bool     altSort       = true;
  bool     sort          = true;
  bool     localPatchCPY = true;
};

GPUWorkingMemory allocateMemory(BVHTest::base::CUDAMemoryBVHPointer *_bvh, uint32_t _batchSize, uint32_t _numFaces);
void             freeMemory(GPUWorkingMemory *_data);

void initData(GPUWorkingMemory *_data, BVHTest::base::CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize);
void fixTree1(GPUWorkingMemory *_data, BVHTest::base::CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize);
void fixTree3(GPUWorkingMemory *_data, BVHTest::base::CUDAMemoryBVHPointer *_GPUbvh, uint32_t _blockSize);

void doAlgorithmStep(GPUWorkingMemory *                   _data,
                     BVHTest::base::CUDAMemoryBVHPointer *_GPUbvh,
                     uint32_t                             _numChunks,
                     uint32_t                             _chunkSize,
                     AlgoCFG                              _cfg);

void doCudaDevSync();

uint32_t calcNumSkipped(GPUWorkingMemory *_data);
