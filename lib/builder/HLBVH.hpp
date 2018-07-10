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

#include "BuilderBase.hpp"
#include "HLBVH_CUDA.hpp"

namespace BVHTest::builder {

class HLBVH final : public BuilderBase {
 private:
  HLBVH_WorkingMemory vWorkingMem;

 public:
  HLBVH() = default;
  virtual ~HLBVH();

  inline std::string getName() const override { return "hlbvh"; }
  inline std::string getDesc() const override { return "HLBVH CUDA builder"; }

  inline uint64_t getRequiredCommands() const override {
    return BuilderBase::getRequiredCommands() | static_cast<uint64_t>(base::CommandType::CUDA_INIT);
  }

  base::ErrorCode setup(base::State &_state) override;
  base::ErrorCode runImpl(base::State &_state) override;
  void            teardown(base::State &_state) override;

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::builder
