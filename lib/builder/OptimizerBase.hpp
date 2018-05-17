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
#include "base/Command.hpp"

namespace BVHTest::builder {

class OptimizerBase : public base::Command {
  double vCostInner = 1.2f;
  double vCostTri   = 1.0f;

 public:
  OptimizerBase() = default;
  virtual ~OptimizerBase();

  inline base::CommandType getType() const override { return base::CommandType::BVH_OPT1; }
  inline uint64_t getRequiredCommands() const override { return static_cast<uint64_t>(base::CommandType::BVH_BUILD); }

  void fromJSON(const json &_j) override;
  json toJSON() const override;

  inline double getCostInner() const noexcept { return vCostInner; }
  inline double getCostTri() const noexcept { return vCostTri; }
};

} // namespace BVHTest::builder
