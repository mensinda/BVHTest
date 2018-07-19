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

namespace BVHTest::builder {

class Bittner13Par final : public OptimizerBase {
 public:
  typedef std::pair<uint32_t, float> TUP;

  struct SumMin {
    float                sum;
    float                min;
    std::atomic_uint32_t flag;
  };

  struct RM_RES {
    struct NodePair {
      uint32_t n1;
      uint32_t n2;
    };

    bool     res;
    NodePair toInsert;
    NodePair unused;
    NodePair etc;
  };

  struct INS_RES {
    bool     res;
    uint32_t best;
    uint32_t root;
  };

  static const size_t QUEUE_SIZE = 16384;

 private:
  uint32_t vMaxNumStepps = 500;
  uint32_t vNumChunks    = 32;
  uint32_t vAltFNQSize   = 16;
  float    vBatchPercent = 1.0f;
  bool     vRandom       = false;
  bool     vSortBatch    = false;
  bool     vOffsetAccess = false;
  bool     vShuffleList  = true;
  bool     vAltFindNode  = false;

  std::vector<TUP>            vTodoList;
  std::vector<base::BVHPatch> vPatches;
  std::vector<bool>           vSkipp;
  std::vector<uint32_t>       vFixList;
  std::unique_ptr<SumMin[]>   vSumMin;

  uint32_t findNode1(uint32_t _n, base::BVHPatch &_bvh);
  uint32_t findNode2(uint32_t _n, base::BVHPatch &_bvh);

  RM_RES  removeNode(uint32_t _node, base::BVHPatch &_bvh, SumMin *_sumMin);
  INS_RES reinsert(uint32_t _node, uint32_t _unused, base::BVHPatch &_bvh, bool _update, SumMin *_sumMin);
  void    fixTree(uint32_t _node, base::BVH &_bvh, SumMin *_sumMin);
  void    initSumAndMin(base::BVH &_bvh, SumMin *_sumMin);

 public:
  Bittner13Par() = default;
  virtual ~Bittner13Par();

  inline std::string getName() const override { return "bittner13Par"; }
  inline std::string getDesc() const override { return "parallel BVH optimizer based on Bittner et al. 2013"; }

  base::ErrorCode setup(base::State &_state) override;
  base::ErrorCode runImpl(base::State &_state) override;
  void            teardown(base::State &_state) override;

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::builder
