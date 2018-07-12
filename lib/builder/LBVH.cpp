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

#include "LBVH.hpp"

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::builder;

LBVH::~LBVH() {}
void LBVH::fromJSON(const json &_j) { BuilderBase::fromJSON(_j); }
json LBVH::toJSON() const { return BuilderBase::toJSON(); }

ErrorCode LBVH::setup(State &_state) {
  vWorkingMem = LBVH_allocateWorkingMemory(&_state.cudaMem.rawMesh);
  if (!vWorkingMem.lRes) { return ErrorCode::CUDA_ERROR; }

  if (!LBVH_allocateBVH(&_state.cudaMem.bvh, &_state.cudaMem.rawMesh)) {
    LBVH_freeWorkingMemory(&vWorkingMem);
    return ErrorCode::CUDA_ERROR;
  }

  (void)_state;
  return ErrorCode::OK;
}

ErrorCode LBVH::runImpl(State &_state) {
  AABB lBBox = LBVH_initTriData(&vWorkingMem, &_state.cudaMem.rawMesh);
  LBVH_calcMortonCodes(&vWorkingMem, lBBox);
  LBVH_sortMortonCodes(&vWorkingMem);
  LBVH_buildBVHTree(&vWorkingMem, &_state.cudaMem.bvh);
  LBVH_fixAABB(&vWorkingMem, &_state.cudaMem.bvh);
  LBVH_doCUDASyc();

  return ErrorCode::OK;
}

void LBVH::teardown(State &_state) {
  (void)_state;
  LBVH_freeWorkingMemory(&vWorkingMem);
}
