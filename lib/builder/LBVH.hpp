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
#include "LBVH_CUDA.hpp"

namespace BVHTest::builder {

class LBVH final : public BuilderBase {
 private:
  LBVH_WorkingMemory vWorkingMem;

 public:
  LBVH() = default;
  virtual ~LBVH();

  inline std::string getName() const override { return "LBVH"; }
  inline std::string getDesc() const override { return "LBVH CUDA builder"; }

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
