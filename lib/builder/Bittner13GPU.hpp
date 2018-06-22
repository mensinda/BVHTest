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

#include "base/BVHPatch.hpp"
#include "Bittner13CUDA.hpp"
#include "OptimizerBase.hpp"
#include <atomic>

namespace BVHTest::builder {

class Bittner13GPU final : public OptimizerBase {
 private:
  uint32_t vMaxNumStepps  = 500;
  uint32_t vNumChunks     = 32;
  uint32_t vCUDABlockSize = 128;
  float    vBatchPercent  = 1.0f;
  bool     vRandom        = false;
  bool     vSortBatch     = false;
  bool     vOffsetAccess  = false;
  bool     vRetryLocking  = true;
  bool     vAltFindNode   = false;
  bool     vAltFixTree    = false;

  GPUWorkingMemory vWorkingMemory;

 public:
  Bittner13GPU() = default;
  virtual ~Bittner13GPU();

  inline std::string getName() const override { return "bittner13GPU"; }
  inline std::string getDesc() const override { return "CUDA: parallel BVH optimizer based on Bittner et al. 2013"; }

  inline uint64_t getRequiredCommands() const override {
    return static_cast<uint64_t>(base::CommandType::BVH_BUILD) | static_cast<uint64_t>(base::CommandType::CUDA_INIT);
  }

  base::ErrorCode setup(base::State &_state) override;
  base::ErrorCode runImpl(base::State &_state) override;
  void            teardown(base::State &_state) override;

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::builder
