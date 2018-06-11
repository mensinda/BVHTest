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

#include "BVHTestCfg.hpp"
#include "Bittner13GPU.hpp"
#include "misc/OMPReductions.hpp"
#include "Bittner13CUDA.hpp"
#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <random>
#include <thread>

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;
using namespace BVHTest::misc;


// Quality of life defines
#define SUM_OF(x) _sumMin[x].sum
#define MIN_OF(x) _sumMin[x].min

#define NODE _bvh[lNode]
#define PARENT _bvh[NODE->parent]
#define LEFT _bvh[NODE->left]
#define RIGHT _bvh[NODE->right]

Bittner13GPU::~Bittner13GPU() {}

void Bittner13GPU::fromJSON(const json &_j) {
  OptimizerBase::fromJSON(_j);
  vMaxNumStepps  = _j.value("maxNumStepps", vMaxNumStepps);
  vNumChunks     = _j.value("numChunks", vNumChunks);
  vCUDABlockSize = _j.value("CUDABlockSize", vCUDABlockSize);
  vBatchPercent  = _j.value("batchPercent", vBatchPercent);
  vRandom        = _j.value("random", vRandom);
  vSortBatch     = _j.value("sort", vSortBatch);
  vShuffleList   = _j.value("shuffle", vShuffleList);
  vOffsetAccess  = _j.value("offsetAccess", vOffsetAccess);

  if (vBatchPercent <= 0.01f) vBatchPercent = 0.01f;
  if (vBatchPercent >= 75.0f) vBatchPercent = 75.0f;
}

json Bittner13GPU::toJSON() const {
  json lJSON             = OptimizerBase::toJSON();
  lJSON["maxNumStepps"]  = vMaxNumStepps;
  lJSON["numChunks"]     = vNumChunks;
  lJSON["CUDABlockSize"] = vCUDABlockSize;
  lJSON["batchPercent"]  = vBatchPercent;
  lJSON["random"]        = vRandom;
  lJSON["sort"]          = vSortBatch;
  lJSON["shuffle"]       = vShuffleList;
  lJSON["offsetAccess"]  = vOffsetAccess;
  return lJSON;
}


ErrorCode Bittner13GPU::runImpl(State &_state) {
  uint32_t lNumNodes        = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  uint32_t lChunkSize       = lNumNodes / vNumChunks;
  lNumNodes                 = lChunkSize * vNumChunks;
  uint32_t         lSkipped = 0;
  GPUWorkingMemory lMem     = allocateMemory(&_state.cudaMem.bvh, lChunkSize, _state.cudaMem.rawMesh.numFaces);

  if (!lMem.result) { return ErrorCode::CUDA_ERROR; }

  initData(&lMem, &_state.cudaMem.bvh, vCUDABlockSize);
  fixTree(&lMem, &_state.cudaMem.bvh, vCUDABlockSize);


  for (uint32_t i = 0; i < vMaxNumStepps; ++i) {
    progress(fmt::format("Stepp {:<3}; SAH: ?", i), i, vMaxNumStepps - 1);

    doAlgorithmStep(&lMem, &_state.cudaMem.bvh, vNumChunks, lChunkSize, vCUDABlockSize, vOffsetAccess);
  }

  PROGRESS_DONE;

  lSkipped = calcNumSkipped(&lMem);

  freeMemory(&lMem);

  getLogger()->info("Skipped {:<8} of {:<8} -- {}%",
                    lSkipped,
                    lNumNodes * vMaxNumStepps,
                    (int)(((float)lSkipped / (float)(lNumNodes * vMaxNumStepps)) * 100));

  return ErrorCode::OK;
}
