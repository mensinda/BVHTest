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
#include <cfloat>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
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


/*   _____      _ _     _____    _______      _          */
/*  |_   _|    (_) |   |_   _|  (_)  _  \    | |         */
/*    | | _ __  _| |_    | |_ __ _| | | |__ _| |_ __ _   */
/*    | || '_ \| | __|   | | '__| | | | / _` | __/ _` |  */
/*   _| || | | | | |_    | | |  | | |/ / (_| | || (_| |  */
/*   \___/_| |_|_|\__|   \_/_|  |_|___/ \__,_|\__\__,_|  */
/*                                                       */
/*                                                       */

struct ReduceRootAABB {
  __device__ __forceinline__ HLBVH_TriData operator()(HLBVH_TriData const &a, HLBVH_TriData const &b) {
    HLBVH_TriData lRes;
    lRes.bbox = a.bbox;
    lRes.bbox.mergeWith(b.bbox);
    return lRes;
  }
};

extern "C" __global__ void kInitTriData(HLBVH_TriData *_data, Triangle *_faces, vec3 *_vert, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) {
    HLBVH_TriData lRes;
    lRes.faceIndex = i;

    Triangle const lFace = _faces[i];
    vec3 const     lV1   = _vert[lFace.v1];
    vec3 const     lV2   = _vert[lFace.v2];
    vec3 const     lV3   = _vert[lFace.v3];

    lRes.bbox.minMax[0] = lV1;
    lRes.bbox.minMax[1] = lV2;

    if (lV2.x < lRes.bbox.minMax[0].x) { lRes.bbox.minMax[0].x = lV2.x; }
    if (lV2.y < lRes.bbox.minMax[0].y) { lRes.bbox.minMax[0].y = lV2.y; }
    if (lV2.z < lRes.bbox.minMax[0].z) { lRes.bbox.minMax[0].z = lV2.z; }
    if (lV3.x < lRes.bbox.minMax[0].x) { lRes.bbox.minMax[0].x = lV3.x; }
    if (lV3.y < lRes.bbox.minMax[0].y) { lRes.bbox.minMax[0].y = lV3.y; }
    if (lV3.z < lRes.bbox.minMax[0].z) { lRes.bbox.minMax[0].z = lV3.z; }

    if (lV1.x > lRes.bbox.minMax[1].x) { lRes.bbox.minMax[1].x = lV1.x; }
    if (lV1.y > lRes.bbox.minMax[1].y) { lRes.bbox.minMax[1].y = lV1.y; }
    if (lV1.z > lRes.bbox.minMax[1].z) { lRes.bbox.minMax[1].z = lV1.z; }
    if (lV3.x > lRes.bbox.minMax[1].x) { lRes.bbox.minMax[1].x = lV3.x; }
    if (lV3.y > lRes.bbox.minMax[1].y) { lRes.bbox.minMax[1].y = lV3.y; }
    if (lV3.z > lRes.bbox.minMax[1].z) { lRes.bbox.minMax[1].z = lV3.z; }

    lRes.bbox.minMax[0] -= FLT_EPSILON;
    lRes.bbox.minMax[1] += FLT_EPSILON;

    lRes.centroid = (lV1 + lV2 + lV3) / 3.0f;

    _data[i] = lRes;
  }
}

AABB HLBVH_initTriData(HLBVH_WorkingMemory *_mem, MeshRaw *_rawMesh) {
  cudaError_t    lRes;
  ReduceRootAABB lReductor;
  uint32_t       lNumBlocks = (_rawMesh->numFaces + 64 - 1) / 64;
  HLBVH_TriData *lDevResult = nullptr;
  HLBVH_TriData  lResult;
  HLBVH_TriData  lInit;
  lInit.bbox.minMax[0] = vec3(0.0f, 0.0f, 0.0f);
  lInit.bbox.minMax[1] = vec3(0.0f, 0.0f, 0.0f);

  ALLOCATE(&lDevResult, 1, HLBVH_TriData);

  kInitTriData<<<lNumBlocks, 64>>>(_mem->triData, _rawMesh->faces, _rawMesh->vert, _rawMesh->numFaces);

  CUDA_RUN(cub::DeviceReduce::Reduce(
      _mem->cubTempStorage, _mem->cubTempStorageSize, _mem->triData, lDevResult, _mem->numFaces, lReductor, lInit));
  CUDA_RUN(cudaMemcpy(&lResult, lDevResult, sizeof(HLBVH_TriData), cudaMemcpyDeviceToHost));

error:
  FREE(lDevResult);
  return lResult.bbox;
}

/*   _____       _       ___  ___           _                _____           _             */
/*  /  __ \     | |      |  \/  |          | |              /  __ \         | |            */
/*  | /  \/ __ _| | ___  | .  . | ___  _ __| |_ ___  _ __   | /  \/ ___   __| | ___  ___   */
/*  | |    / _` | |/ __| | |\/| |/ _ \| '__| __/ _ \| '_ \  | |    / _ \ / _` |/ _ \/ __|  */
/*  | \__/\ (_| | | (__  | |  | | (_) | |  | || (_) | | | | | \__/\ (_) | (_| |  __/\__ \  */
/*   \____/\__,_|_|\___| \_|  |_/\___/|_|   \__\___/|_| |_|  \____/\___/ \__,_|\___||___/  */
/*                                                                                         */
/*                                                                                         */

// Expands a 10-bit integer into 30 bits
// by inserting 2 zeros after each bit.
__device__ __forceinline__ uint32_t expandBits(uint32_t v) {
  v = (v * 0x00010001u) & 0xFF0000FFu;
  v = (v * 0x00000101u) & 0x0F00F00Fu;
  v = (v * 0x00000011u) & 0xC30C30C3u;
  v = (v * 0x00000005u) & 0x49249249u;
  return v;
}

extern "C" __global__ void kCalcMortonCodes(
    uint32_t *_outCode, HLBVH_TriData *_triData, vec3 _offset, vec3 _scale, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) {
    vec3 lCentroid = _triData[i].centroid;

    // normalize controid to range [0, 1]
    lCentroid += _offset;
    lCentroid *= _scale;

    // calc morton code
    lCentroid *= 1024.0f;
    if (lCentroid.x < 0.0f) { lCentroid.x = 0.0f; }
    if (lCentroid.y < 0.0f) { lCentroid.y = 0.0f; }
    if (lCentroid.z < 0.0f) { lCentroid.z = 0.0f; }
    if (lCentroid.x > 1023.0f) { lCentroid.x = 1023.0f; }
    if (lCentroid.y > 1023.0f) { lCentroid.y = 1023.0f; }
    if (lCentroid.z > 1023.0f) { lCentroid.z = 1023.0f; }

    uint32_t x = expandBits(static_cast<uint32_t>(lCentroid.x));
    uint32_t y = expandBits(static_cast<uint32_t>(lCentroid.y));
    uint32_t z = expandBits(static_cast<uint32_t>(lCentroid.z));

    _outCode[i] = x * 4 + y * 2 + z;
  }
}

void HLBVH_calcMortonCodes(HLBVH_WorkingMemory *_mem, AABB _sceneAABB) {
  uint32_t lNumBlocks = (_mem->numFaces + 64 - 1) / 64;

  // Set AABB to range [0, X]
  vec3 lOffset = vec3(0.0f, 0.0f, 0.0f) - _sceneAABB.minMax[0];
  _sceneAABB.minMax[0] += lOffset;
  _sceneAABB.minMax[1] += lOffset;

  // Calculate scale sot that AABB is in range [0, 1]
  vec3 lScale;
  lScale.x = 1.0f / _sceneAABB.minMax[1].x;
  lScale.y = 1.0f / _sceneAABB.minMax[1].y;
  lScale.z = 1.0f / _sceneAABB.minMax[1].z;

  kCalcMortonCodes<<<lNumBlocks, 64>>>(_mem->mortonCodes, _mem->triData, lOffset, lScale, _mem->numFaces);
}

void HLBVH_sortMortonCodes(HLBVH_WorkingMemory *_mem) {
  cub::DeviceRadixSort::SortPairs(_mem->cubTempStorage,
                                  _mem->cubTempStorageSize,
                                  _mem->mortonCodes,
                                  _mem->mortonCodesSorted,
                                  _mem->triData,
                                  _mem->triDataSorted,
                                  _mem->numFaces);
}

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
  cudaError_t         lRes;
  HLBVH_WorkingMemory lMem;
  ReduceRootAABB      lReductor;
  size_t              lTS1   = 0;
  size_t              lTS2   = 0;
  HLBVH_TriData *     lDummy = nullptr;
  HLBVH_TriData       lInit;
  lInit.bbox.minMax[0] = vec3(0.0f, 0.0f, 0.0f);
  lInit.bbox.minMax[1] = vec3(0.0f, 0.0f, 0.0f);

  lMem.numFaces = _rawMesh->numFaces;
  ALLOCATE(&lMem.mortonCodes, lMem.numFaces, uint32_t);
  ALLOCATE(&lMem.mortonCodesSorted, lMem.numFaces, uint32_t);
  ALLOCATE(&lMem.triData, lMem.numFaces, HLBVH_TriData);
  ALLOCATE(&lMem.triDataSorted, lMem.numFaces, HLBVH_TriData);

  CUDA_RUN(cub::DeviceRadixSort::SortPairs(lMem.cubTempStorage,
                                           lTS1,
                                           lMem.mortonCodes,
                                           lMem.mortonCodesSorted,
                                           lMem.triData,
                                           lMem.triDataSorted,
                                           lMem.numFaces));

  CUDA_RUN(cub::DeviceReduce::Reduce(lMem.cubTempStorage, lTS2, lMem.triData, lDummy, lMem.numFaces, lReductor, lInit));

  lMem.cubTempStorageSize = lTS1 > lTS2 ? lTS1 : lTS2;

  ALLOCATE(&lMem.cubTempStorage, lMem.cubTempStorageSize, uint8_t);

  lMem.lRes = true;
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
