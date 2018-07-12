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

using BVHTest::base::AABB;
using BVHTest::base::CUDAMemoryBVHPointer;
using BVHTest::base::MeshRaw;
using BVHTest::base::Triangle;
using glm::vec3;

struct LBVH_TriData {
  AABB     bbox;
  vec3     centroid;
  uint32_t faceIndex;
  uint64_t __padding__;
};

struct LBVH_WorkingMemory {
  bool lRes = false;

  uint32_t *    mortonCodes       = nullptr;
  uint32_t *    mortonCodesSorted = nullptr;
  LBVH_TriData *triData           = nullptr;
  LBVH_TriData *triDataSorted     = nullptr;
  void *        cubTempStorage    = nullptr;
  uint32_t *    atomicLocks       = nullptr;

  uint32_t numFaces           = 0;
  size_t   cubTempStorageSize = 0;
  uint32_t numLocks           = 0;
};

AABB LBVH_initTriData(LBVH_WorkingMemory *_mem, MeshRaw *_rawMesh);
void LBVH_calcMortonCodes(LBVH_WorkingMemory *_mem, AABB _sceneAABB);
void LBVH_sortMortonCodes(LBVH_WorkingMemory *_mem);
void LBVH_buildBVHTree(LBVH_WorkingMemory *_mem, CUDAMemoryBVHPointer *_bvh);
void LBVH_fixAABB(LBVH_WorkingMemory *_mem, CUDAMemoryBVHPointer *_bvh);

bool               LBVH_allocateBVH(CUDAMemoryBVHPointer *_bvh, MeshRaw *_rawMesh);
LBVH_WorkingMemory LBVH_allocateWorkingMemory(MeshRaw *_rawMesh);
void               LBVH_freeWorkingMemory(LBVH_WorkingMemory *_mem);

void LBVH_doCUDASyc();
