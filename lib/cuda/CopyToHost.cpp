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

#include "CopyToHost.hpp"
#include "cudaFN.hpp"

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::cuda;

CopyToHost::~CopyToHost() {}

void CopyToHost::fromJSON(const json &_j) { vFixLevels = _j.value("fixLevels", vFixLevels); }
json CopyToHost::toJSON() const { return {{"fixLevels", vFixLevels}}; }

ErrorCode CopyToHost::runImpl(State &_state) {
  if (!copyBVHToHost(&_state.cudaMem.bvh, &_state.bvh)) { return ErrorCode::CUDA_ERROR; }
  if (!copyMeshToHost(&_state.cudaMem.rawMesh, &_state.mesh)) { return ErrorCode::CUDA_ERROR; }
  if (vFixLevels) { _state.bvh.fixLevels(); }
  return ErrorCode::OK;
}
