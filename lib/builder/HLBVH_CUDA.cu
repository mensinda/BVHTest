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

#include "HLBVH_CUDA.hpp"
#include <cub/cub.cuh>
#include <cuda_runtime_api.h>
#include <iostream>

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::base;

#define CUDA_RUN(call)                                                                                                 \
  lRes = call;                                                                                                         \
  if (lRes != cudaSuccess) {                                                                                           \
    cout << "CUDA ERROR (" << __FILE__ << ":" << __LINE__ << "): " << cudaGetErrorString(lRes) << endl;                \
    goto error;                                                                                                        \
  }

#define ALLOCATE(ptr, num, type) CUDA_RUN(cudaMalloc(ptr, num * sizeof(type)));
#define FREE(ptr)                                                                                                      \
  cudaFree(ptr);                                                                                                       \
  ptr = nullptr;



/*  ___  ___                                 ___  ___                                                  _     */
/*  |  \/  |                                 |  \/  |                                                 | |    */
/*  | .  . | ___ _ __ ___   ___  _ __ _   _  | .  . | __ _ _ __   __ _  __ _  ___ _ __ ___   ___ _ __ | |_   */
/*  | |\/| |/ _ \ '_ ` _ \ / _ \| '__| | | | | |\/| |/ _` | '_ \ / _` |/ _` |/ _ \ '_ ` _ \ / _ \ '_ \| __|  */
/*  | |  | |  __/ | | | | | (_) | |  | |_| | | |  | | (_| | | | | (_| | (_| |  __/ | | | | |  __/ | | | |_   */
/*  \_|  |_/\___|_| |_| |_|\___/|_|   \__, | \_|  |_/\__,_|_| |_|\__,_|\__, |\___|_| |_| |_|\___|_| |_|\__|  */
/*                                     __/ |                            __/ |                                */
/*                                    |___/                            |___/                                 */

bool HLBVH_allocateBVH(CUDAMemoryBVHPointer *_bvh, MeshRaw *_rawMesh) {
  cudaError_t lRes;

  _bvh->numNodes = _rawMesh->numFaces * 2;
  ALLOCATE(&_bvh->bvh, 1, BVH);
  ALLOCATE(&_bvh->nodes, _bvh->numNodes, BVHNode);

  return true;

error:
  FREE(_bvh->nodes);
  FREE(_bvh->bvh);

  _bvh->numNodes = 0;

  return false;
}

HLBVH_WorkingMemory HLBVH_allocateWorkingMemory(MeshRaw *_rawMesh) {
  HLBVH_WorkingMemory lMem;

  cudaError_t lRes;
  lMem.numFaces = _rawMesh->numFaces;
  ALLOCATE(&lMem.mortonCodes, lMem.numFaces, uint32_t);
  ALLOCATE(&lMem.mortonCodesSorted, lMem.numFaces, uint32_t);
  ALLOCATE(&lMem.triData, lMem.numFaces, HLBVH_TriData);
  ALLOCATE(&lMem.triDataSorted, lMem.numFaces, HLBVH_TriData);

  CUDA_RUN(cub::DeviceRadixSort::SortPairs(lMem.cubTempStorage,
                                           lMem.cubTempStorageSize,
                                           lMem.mortonCodes,
                                           lMem.mortonCodesSorted,
                                           lMem.triData,
                                           lMem.triDataSorted,
                                           lMem.numFaces));

  ALLOCATE(&lMem.cubTempStorage, lMem.cubTempStorageSize, uint8_t);

  return lMem;

error:
  lMem.lRes               = false;
  lMem.numFaces           = 0;
  lMem.cubTempStorageSize = 0;

  FREE(lMem.mortonCodes);
  FREE(lMem.mortonCodesSorted);
  FREE(lMem.triData);
  FREE(lMem.triDataSorted);
  FREE(lMem.cubTempStorage);

  return lMem;
}

void HLBVH_freeWorkingMemory(HLBVH_WorkingMemory *_mem) {
  FREE(_mem->mortonCodes);
  FREE(_mem->mortonCodesSorted);
  FREE(_mem->triData);
  FREE(_mem->triDataSorted);
  FREE(_mem->cubTempStorage);

  _mem->cubTempStorageSize = 0;
  _mem->numFaces           = 0;
  _mem->lRes               = false;
}
