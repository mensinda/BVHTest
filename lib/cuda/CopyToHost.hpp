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

#include "base/Command.hpp"

namespace BVHTest::cuda {

class CopyToHost final : public base::Command {
 public:
  CopyToHost() = default;
  virtual ~CopyToHost();

  inline std::string getName() const override { return "copyToHost"; }
  std::string        getDesc() const override { return "CUDA: copies data from GPU to main memory (frees GPU memory)"; }
  base::CommandType  getType() const override { return base::CommandType::CUDA_DONE; }
  uint64_t getRequiredCommands() const override { return static_cast<uint64_t>(base::CommandType::CUDA_INIT); }

  base::ErrorCode runImpl(base::State &_state) override;

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::cuda
