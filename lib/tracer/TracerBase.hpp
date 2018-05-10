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

namespace BVHTest::tracer {

class TracerBase : public base::Command {
 public:
  TracerBase() = default;
  virtual ~TracerBase();

  inline base::CommandType getType() const override { return base::CommandType::RAY_TRACE; }
  inline uint64_t getRequiredCommands() const override { return static_cast<uint64_t>(base::CommandType::BVH_BUILD); }
};

} // namespace BVHTest::tracer
