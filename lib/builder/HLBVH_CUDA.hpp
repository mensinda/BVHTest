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

struct HLBVH_TriData {
  AABB     bbox;
  vec3     centroid;
  uint32_t faceIndex;
  uint64_t __padding__;
};

struct HLBVH_WorkingMemory {
  bool lRes = false;

  uint32_t *     mortonCodes       = nullptr;
  uint32_t *     mortonCodesSorted = nullptr;
  HLBVH_TriData *triData           = nullptr;
  HLBVH_TriData *triDataSorted     = nullptr;
  void *         cubTempStorage    = nullptr;

  uint32_t numFaces           = 0;
  size_t   cubTempStorageSize = 0;
};

AABB HLBVH_initTriData(HLBVH_WorkingMemory *_mem, MeshRaw *_rawMesh);
void HLBVH_calcMortonCodes(HLBVH_WorkingMemory *_mem, AABB _sceneAABB);

bool                HLBVH_allocateBVH(CUDAMemoryBVHPointer *_bvh, MeshRaw *_rawMesh);
HLBVH_WorkingMemory HLBVH_allocateWorkingMemory(MeshRaw *_rawMesh);
void                HLBVH_freeWorkingMemory(HLBVH_WorkingMemory *_mem);
