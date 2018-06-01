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
#include "OptimizerBase.hpp"
#include <atomic>
#include <tuple>

namespace BVHTest::builder {

class Bittner13Par final : public OptimizerBase {
 public:
  typedef base::BVHPatch<10, 2, 128> PATCH;
  static_assert(sizeof(std::atomic_uint32_t) == sizeof(uint32_t), "Size of atomic");
  typedef std::tuple<float, float, std::atomic_uint32_t>                                             SumMin;
  typedef std::tuple<bool, std::tuple<uint32_t, uint32_t>, std::tuple<uint32_t, uint32_t>, uint32_t> RM_RES;

 private:
  uint32_t vMaxNumStepps = 500;
  float    vBatchPercent = 1.0f;
  bool     vRandom       = false;
  bool     vSortBatch    = true;

  std::pair<uint32_t, uint32_t> findNodeForReinsertion(uint32_t _n, PATCH &_bvh);

  RM_RES removeNode(uint32_t _node, PATCH &_bvh, SumMin *_sumMin);
  bool   reinsert(uint32_t _node, uint32_t _unused, PATCH &_bvh, bool _update, SumMin *_sumMin);
  void   fixTree(uint32_t _node, base::BVH &_bvh, SumMin *_sumMin);
  void   initSumAndMin(base::BVH &_bvh, SumMin *_sumMin);

 public:
  Bittner13Par() = default;
  virtual ~Bittner13Par();

  inline std::string getName() const override { return "bittner13Par"; }
  inline std::string getDesc() const override { return "parallel BVH optimizer based on Bittner et al. 2013"; }

  base::ErrorCode runImpl(base::State &_state) override;

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::builder
