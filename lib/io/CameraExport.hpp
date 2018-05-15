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

namespace BVHTest::IO {

class CameraExport final : public base::Command {
  const uint32_t vFormatVers = 1;

 public:
  CameraExport() = default;
  virtual ~CameraExport();

  inline std::string getName() const override { return "camExport"; }
  inline std::string getDesc() const override { return "Write generated cameras to a JSON file"; }
  inline uint64_t    getRequiredCommands() const override { return static_cast<uint64_t>(base::CommandType::IMPORT); }
  inline base::CommandType getType() const override { return base::CommandType::EXPORT; }

  base::ErrorCode runImpl(base::State &_state) override;

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::IO
