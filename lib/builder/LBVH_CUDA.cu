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

#include "LBVH_CUDA.hpp"
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
  __device__ __forceinline__ LBVH_TriData operator()(LBVH_TriData const &a, LBVH_TriData const &b) {
    LBVH_TriData lRes;
    lRes.bbox = a.bbox;
    lRes.bbox.mergeWith(b.bbox);
    return lRes;
  }
};

extern "C" __global__ void kInitTriData(LBVH_TriData *_data, Triangle *_faces, vec3 *_vert, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) {
    LBVH_TriData lRes;
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

AABB LBVH_initTriData(LBVH_WorkingMemory *_mem, MeshRaw *_rawMesh) {
  cudaError_t    lRes;
  ReduceRootAABB lReductor;
  uint32_t       lNumBlocks = (_rawMesh->numFaces + 64 - 1) / 64;
  LBVH_TriData * lDevResult = nullptr;
  LBVH_TriData   lResult;
  LBVH_TriData   lInit;
  lInit.bbox.minMax[0] = vec3(0.0f, 0.0f, 0.0f);
  lInit.bbox.minMax[1] = vec3(0.0f, 0.0f, 0.0f);

  ALLOCATE(&lDevResult, 1, LBVH_TriData);

  kInitTriData<<<lNumBlocks, 64>>>(_mem->triData, _rawMesh->faces, _rawMesh->vert, _rawMesh->numFaces);

  CUDA_RUN(cub::DeviceReduce::Reduce(
      _mem->cubTempStorage, _mem->cubTempStorageSize, _mem->triData, lDevResult, _mem->numFaces, lReductor, lInit));
  CUDA_RUN(cudaMemcpy(&lResult, lDevResult, sizeof(LBVH_TriData), cudaMemcpyDeviceToHost));

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
    uint32_t *_outCode, LBVH_TriData *_triData, vec3 _offset, vec3 _scale, uint32_t _num) {
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

void LBVH_calcMortonCodes(LBVH_WorkingMemory *_mem, AABB _sceneAABB) {
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

void LBVH_sortMortonCodes(LBVH_WorkingMemory *_mem) {
  cub::DeviceRadixSort::SortPairs(_mem->cubTempStorage,
                                  _mem->cubTempStorageSize,
                                  _mem->mortonCodes,
                                  _mem->mortonCodesSorted,
                                  _mem->triData,
                                  _mem->triDataSorted,
                                  _mem->numFaces);
}


/*  ______       _ _     _   _____               */
/*  | ___ \     (_) |   | | |_   _|              */
/*  | |_/ /_   _ _| | __| |   | |_ __ ___  ___   */
/*  | ___ \ | | | | |/ _` |   | | '__/ _ \/ _ \  */
/*  | |_/ / |_| | | | (_| |   | | | |  __/  __/  */
/*  \____/ \__,_|_|_|\__,_|   \_/_|  \___|\___|  */
/*                                               */
/*                                               */

#define CODE(x) ((static_cast<uint64_t>(__ldg(&_sortedMortonCodes[x])) << 32) | static_cast<uint64_t>(x))
#define DELTA(I, J) delta(_sortedMortonCodes, _numFaces, I, J)

__device__ __forceinline__ int32_t delta(uint32_t *_sortedMortonCodes, uint32_t _numFaces, uint32_t i, uint32_t j) {
  if (j >= _numFaces) { return -1; }

  return __clz(CODE(i) ^ CODE(j));
}

__device__ __forceinline__ uint32_t findSplit(uint32_t *_sortedMortonCodes, uint32_t _first, uint32_t _last) {
  // Identical Morton codes => split the range in the middle.

  uint64_t firstCode = CODE(_first);
  uint64_t lastCode  = CODE(_last);

  if (firstCode == lastCode) return (_first + _last) >> 1;

  // Calculate the number of highest bits that are the same
  // for all objects, using the count-leading-zeros intrinsic.

  uint32_t commonPrefix = __clz(firstCode ^ lastCode);

  // Use binary search to find where the next bit differs.
  // Specifically, we are looking for the highest object that
  // shares more than commonPrefix bits with the first one.

  uint32_t split = _first; // initial guess
  uint32_t step  = _last - _first;

  do {
    step              = (step + 1) >> 1; // exponential decrease
    uint32_t newSplit = split + step;    // proposed new position

    if (newSplit < _last) {
      uint64_t splitCode   = CODE(newSplit);
      uint32_t splitPrefix = __clz(firstCode ^ splitCode);
      if (splitPrefix > commonPrefix) { split = newSplit; } // accept proposal
    }
  } while (step > 1);

  return split;
}


__device__ __forceinline__ uint2 determineRange(uint32_t *_sortedMortonCodes, uint32_t _numFaces, uint32_t i) {
  int32_t d = DELTA(i, i + 1) - DELTA(i, i - 1);
  assert(d != 0);
  d = (d > 0) ? 1 : -1;

  int32_t  deltaMin = DELTA(i, i - d);
  uint32_t lMax     = 2;

  while (DELTA(i, i + lMax * d) > deltaMin) { lMax *= 2; }
  uint32_t l = 0;

  do {
    lMax = lMax >> 1; // exponential decrease
    if (DELTA(i, i + (l + lMax) * d) > deltaMin) { l += lMax; }
  } while (lMax > 1);

  uint32_t j = i + l * d;
  if (i < j) {
    return {i, j};
  } else {
    return {j, i};
  }
}

extern "C" __global__ void kGenLeafNodes(BVHNode *     _nodes,
                                         uint32_t *    _sortedMortonCodes,
                                         LBVH_TriData *_sortedData,
                                         uint32_t      _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < _num; i += stride) {
    LBVH_TriData lData = _sortedData[i];
    BVHNode      lNode;
    lNode.bbox        = lData.bbox;
    lNode.parent      = UINT32_MAX;
    lNode.numChildren = 0;
    lNode.left        = lData.faceIndex;
    lNode.right       = 1;
    lNode.level       = UINT8_MAX;
    lNode.surfaceArea = lNode.bbox.surfaceArea();

    _nodes[_num - 1 + i] = lNode;
  }
}

extern "C" __global__ void kBuildTree(BVHNode *_nodes, uint32_t *_sortedMortonCodes, uint32_t _numFaces) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (uint32_t i = index; i < (_numFaces - 1); i += stride) {
    // Find out which range of objects the node corresponds to.
    // (This is where the magic happens!)

    uint2    range = determineRange(_sortedMortonCodes, _numFaces, i);
    uint32_t first = range.x;
    uint32_t last  = range.y;

    // Determine where to split the range.

    uint32_t split = findSplit(_sortedMortonCodes, first, last);

    // Select childA and childB.

    uint32_t childAIndex = split;
    uint32_t childBIndex = split + 1;
    if (childAIndex == first) { childAIndex += _numFaces - 1; }
    if (childBIndex == last) { childBIndex += _numFaces - 1; }

    _nodes[i].left             = childAIndex;
    _nodes[i].right            = childBIndex;
    _nodes[childAIndex].parent = i;
    _nodes[childAIndex].isLeft = TRUE;
    _nodes[childBIndex].parent = i;
    _nodes[childBIndex].isLeft = FALSE;
  }
}

void LBVH_buildBVHTree(LBVH_WorkingMemory *_mem, CUDAMemoryBVHPointer *_bvh) {
  uint32_t lNumBlocks = (_mem->numFaces + 64 - 1) / 64;
  kGenLeafNodes<<<lNumBlocks, 64>>>(_bvh->nodes, _mem->mortonCodesSorted, _mem->triDataSorted, _mem->numFaces);
  kBuildTree<<<lNumBlocks, 64>>>(_bvh->nodes, _mem->mortonCodesSorted, _mem->numFaces);
}

/*   _____                           _          ___    ___  ____________   */
/*  |  __ \                         | |        / _ \  / _ \ | ___ \ ___ \  */
/*  | |  \/ ___ _ __   ___ _ __ __ _| |_ ___  / /_\ \/ /_\ \| |_/ / |_/ /  */
/*  | | __ / _ \ '_ \ / _ \ '__/ _` | __/ _ \ |  _  ||  _  || ___ \ ___ \  */
/*  | |_\ \  __/ | | |  __/ | | (_| | ||  __/ | | | || | | || |_/ / |_/ /  */
/*   \____/\___|_| |_|\___|_|  \__,_|\__\___| \_| |_/\_| |_/\____/\____/   */
/*                                                                         */
/*                                                                         */

extern "C" __global__ void kFixAABBTree(BVHNode *_nodes, uint32_t *_locks, uint32_t _num) {
  uint32_t index  = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;

  uint32_t lNode;
  uint32_t lLeft;
  uint32_t lRight;
  float    lSArea;
  AABB     lAABB;

  for (uint32_t i = index + _num - 1; i < (_num * 2 - 1); i += stride) {
    lNode = _nodes[i].parent;

    while (true) {
      uint32_t lOldLock = atomicCAS(&_locks[lNode], 0, 1);

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

      // Check if root
      if (lNode == _nodes[lNode].parent) { break; }
      lNode = _nodes[lNode].parent;
    }
  }
}

void LBVH_fixAABB(LBVH_WorkingMemory *_mem, CUDAMemoryBVHPointer *_bvh) {
  cudaError_t lRes;
  uint32_t    lNumBlocks = (_mem->numFaces + 64 - 1) / 64;

  CUDA_RUN(cudaMemset(_mem->atomicLocks, 0, _mem->numLocks * sizeof(uint32_t)));
  kFixAABBTree<<<lNumBlocks, 64>>>(_bvh->nodes, _mem->atomicLocks, _mem->numFaces);

error:
  return;
}

/*  ___  ___                                 ___  ___                                                  _     */
/*  |  \/  |                                 |  \/  |                                                 | |    */
/*  | .  . | ___ _ __ ___   ___  _ __ _   _  | .  . | __ _ _ __   __ _  __ _  ___ _ __ ___   ___ _ __ | |_   */
/*  | |\/| |/ _ \ '_ ` _ \ / _ \| '__| | | | | |\/| |/ _` | '_ \ / _` |/ _` |/ _ \ '_ ` _ \ / _ \ '_ \| __|  */
/*  | |  | |  __/ | | | | | (_) | |  | |_| | | |  | | (_| | | | | (_| | (_| |  __/ | | | | |  __/ | | | |_   */
/*  \_|  |_/\___|_| |_| |_|\___/|_|   \__, | \_|  |_/\__,_|_| |_|\__,_|\__, |\___|_| |_| |_|\___|_| |_|\__|  */
/*                                     __/ |                            __/ |                                */
/*                                    |___/                            |___/                                 */

bool LBVH_allocateBVH(CUDAMemoryBVHPointer *_bvh, MeshRaw *_rawMesh) {
  cudaError_t lRes;
  BVH         lBVH;

  _bvh->numNodes = _rawMesh->numFaces * 2 - 1;
  ALLOCATE(&_bvh->bvh, 1, BVH);
  ALLOCATE(&_bvh->nodes, _bvh->numNodes, BVHNode);

  lBVH.setNewRoot(0);
  lBVH.setMemory(_bvh->nodes, _bvh->numNodes, _bvh->numNodes);

  CUDA_RUN(cudaMemcpy(_bvh->bvh, &lBVH, sizeof(BVH), cudaMemcpyHostToDevice));

  lBVH.setMemory(nullptr, 0, 0);

  return true;

error:
  FREE(_bvh->nodes);
  FREE(_bvh->bvh);

  _bvh->numNodes = 0;

  lBVH.setMemory(nullptr, 0, 0);
  return false;
}

LBVH_WorkingMemory LBVH_allocateWorkingMemory(MeshRaw *_rawMesh) {
  cudaError_t        lRes;
  LBVH_WorkingMemory lMem;
  ReduceRootAABB     lReductor;
  size_t             lTS1   = 0;
  size_t             lTS2   = 0;
  LBVH_TriData *     lDummy = nullptr;
  LBVH_TriData       lInit;
  lInit.bbox.minMax[0] = vec3(0.0f, 0.0f, 0.0f);
  lInit.bbox.minMax[1] = vec3(0.0f, 0.0f, 0.0f);

  lMem.numFaces = _rawMesh->numFaces;
  lMem.numLocks = _rawMesh->numFaces * 2 - 1;
  ALLOCATE(&lMem.mortonCodes, lMem.numFaces, uint32_t);
  ALLOCATE(&lMem.mortonCodesSorted, lMem.numFaces, uint32_t);
  ALLOCATE(&lMem.triData, lMem.numFaces, LBVH_TriData);
  ALLOCATE(&lMem.triDataSorted, lMem.numFaces, LBVH_TriData);
  ALLOCATE(&lMem.atomicLocks, lMem.numLocks, uint32_t);

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
  FREE(lMem.atomicLocks);

  return lMem;
}

void LBVH_freeWorkingMemory(LBVH_WorkingMemory *_mem) {
  FREE(_mem->mortonCodes);
  FREE(_mem->mortonCodesSorted);
  FREE(_mem->triData);
  FREE(_mem->triDataSorted);
  FREE(_mem->cubTempStorage);
  FREE(_mem->atomicLocks);

  _mem->cubTempStorageSize = 0;
  _mem->numFaces           = 0;
  _mem->lRes               = false;
}

void LBVH_doCUDASyc() { cudaDeviceSynchronize(); }
