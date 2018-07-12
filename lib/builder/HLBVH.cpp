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

#include "HLBVH.hpp"

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::builder;

HLBVH::~HLBVH() {}
void HLBVH::fromJSON(const json &_j) { BuilderBase::fromJSON(_j); }
json HLBVH::toJSON() const { return BuilderBase::toJSON(); }

ErrorCode HLBVH::setup(State &_state) {
  vWorkingMem = HLBVH_allocateWorkingMemory(&_state.cudaMem.rawMesh);
  if (!vWorkingMem.lRes) { return ErrorCode::CUDA_ERROR; }

  if (!HLBVH_allocateBVH(&_state.cudaMem.bvh, &_state.cudaMem.rawMesh)) {
    HLBVH_freeWorkingMemory(&vWorkingMem);
    return ErrorCode::CUDA_ERROR;
  }

  (void)_state;
  return ErrorCode::OK;
}

ErrorCode HLBVH::runImpl(State &_state) {
  AABB lBBox = HLBVH_initTriData(&vWorkingMem, &_state.cudaMem.rawMesh);
  HLBVH_calcMortonCodes(&vWorkingMem, lBBox);
  HLBVH_sortMortonCodes(&vWorkingMem);
  HLBVH_buildBVHTree(&vWorkingMem, &_state.cudaMem.bvh);
  HLBVH_fixAABB(&vWorkingMem, &_state.cudaMem.bvh);
  HLBVH_doCUDASyc();

  return ErrorCode::OK;
}

void HLBVH::teardown(State &_state) {
  (void)_state;
  HLBVH_freeWorkingMemory(&vWorkingMem);
}
