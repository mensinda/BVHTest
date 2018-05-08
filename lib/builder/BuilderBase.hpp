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

class BuilderBase : public base::Command {
 private:
  float vCostInner = 1.2f;
  float vCostTri   = 1.0f;

 public:
  BuilderBase() = default;
  virtual ~BuilderBase();

  inline base::CommandType getType() const override { return base::CommandType::BVH_BUILD; }
  inline uint64_t getRequiredCommands() const override { return static_cast<uint64_t>(base::CommandType::IMPORT); }

  std::vector<base::TriWithBB> boundingVolumesFromMesh(base::Mesh const &_mesh);

  inline float getCostInner() const noexcept { return vCostInner; }
  inline float getCostTri() const noexcept { return vCostTri; }
};

} // namespace BVHTest::builder
