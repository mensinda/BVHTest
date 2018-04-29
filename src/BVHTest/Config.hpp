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

#include "base/Configurable.hpp"
#include <string>

namespace BVHTest {

class Config final : public base::Configurable {
  std::string name = "BVHTest default run";

 public:
  Config() = default;
  ~Config();

  void fromJSON(json const& _j)  override;
  json toJSON() const override;
};

void to_json(json &_j, const Config &_cfg)  { _j = _cfg.toJSON(); }
void from_json(const json &_j, Config &_cfg) { _cfg.fromJSON(_j); }

} // namespace BVHTest
