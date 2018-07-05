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
 public:
  typedef base::TriWithBB             TYPE;
  typedef TYPE const &                TCREF;
  typedef std::vector<TYPE>::iterator ITER;

  struct BuildRes {
    uint32_t   vecPtr;
    base::AABB bbox;
  };

 private:
  double vCostInner = 1.2f;
  double vCostTri   = 1.0f;

 protected:
  virtual ITER split(ITER _begin, ITER _end, uint32_t _level);

  BuildRes build(ITER                _begin,
                 ITER                _end,
                 BVHTest::base::BVH &_bvh,
                 uint32_t            _parent      = 0,
                 bool                _isLeftChild = 0,
                 uint32_t            _level       = 0);

 public:
  BuilderBase() = default;
  virtual ~BuilderBase();

  inline base::CommandType getType() const override { return base::CommandType::BVH_BUILD; }
  inline uint64_t getRequiredCommands() const override { return static_cast<uint64_t>(base::CommandType::IMPORT); }

  std::vector<base::TriWithBB> boundingVolumesFromMesh(base::Mesh const &_mesh);

  void fromJSON(const json &_j) override;
  json toJSON() const override;

  inline double getCostInner() const noexcept { return vCostInner; }
  inline double getCostTri() const noexcept { return vCostTri; }
};

} // namespace BVHTest::builder
