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

#include "Command.hpp"
#include "Configurable.hpp"
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace BVHTest::base {

class Config final : public base::Configurable {
 public:
  typedef std::shared_ptr<base::Command>      CMD_PTR;
  typedef std::function<CMD_PTR(std::string)> FUNC_PTR;

 private:
  FUNC_PTR commandFromString;

  std::string vName       = "BVHTest default run";
  uint32_t    vMaxThreads = 4;
  bool        vVerbose    = false;

  std::vector<CMD_PTR> vCommands;

 public:
  Config() = delete;
  Config(FUNC_PTR _fromString) : commandFromString(_fromString) {}
  ~Config();

  void fromJSON(json const &_j) override;
  json toJSON() const override;
};

inline void to_json(json &_j, const Config &_cfg) { _j = _cfg.toJSON(); }
inline void from_json(const json &_j, Config &_cfg) { _cfg.fromJSON(_j); }

} // namespace BVHTest::base
