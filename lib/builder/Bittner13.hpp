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

#include "OptimizerBase.hpp"

namespace BVHTest::builder {

class Bittner13 final : public OptimizerBase {
 private:
  uint32_t vMaxNumStepps = 500;
  float    vBatchPercent = 1.0f;

  inline float directCost(base::BVHNode const &_l, base::BVHNode const &_x) {
    base::AABB lMerge = _l.bbox;
    lMerge.mergeWith(_x.bbox);
    return lMerge.surfaceArea();
  }

  uint32_t findNodeForReinsertion(uint32_t _n, base::BVH &_bvh);
  void     reinsert(uint32_t _node, uint32_t _unused, base::BVH &_bvh);
  void     fixBBOX(uint32_t _node, base::BVH &_bvh);
  float    mComb(uint32_t _n, base::BVH &_bvh);

 public:
  Bittner13() = default;
  virtual ~Bittner13();

  inline std::string getName() const override { return "bittner13"; }
  inline std::string getDesc() const override { return "BVH optimizer based on Bittner et al. 2013"; }

  base::ErrorCode runImpl(base::State &_state) override;

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::builder
