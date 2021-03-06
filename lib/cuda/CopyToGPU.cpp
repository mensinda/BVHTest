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

#include "CopyToGPU.hpp"
#include "cudaFN.hpp"

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::cuda;

CopyToGPU::~CopyToGPU() {}

void CopyToGPU::fromJSON(const json &_j) { (void)_j; }
json CopyToGPU::toJSON() const { return json::object(); }

ErrorCode CopyToGPU::runImpl(State &_state) {
  if (!copyBVHToGPU(&_state.bvh, &_state.cudaMem.bvh)) { return ErrorCode::CUDA_ERROR; }
  if (!copyMeshToGPU(&_state.mesh, &_state.cudaMem.rawMesh)) { return ErrorCode::CUDA_ERROR; }
  return ErrorCode::OK;
}
