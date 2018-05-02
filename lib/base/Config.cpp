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

#include "BVHTestCfg.hpp"
#include "Config.hpp"
#include <iostream>

using namespace BVHTest::base;
using namespace std;

Config::~Config() {}

void Config::fromJSON(json const &_j) {
  auto lLogger = getLogger();
  if (_j.count("baseConfig") > 0) {
    vName       = _j["baseConfig"].value("name", vName);
    vMaxThreads = _j["baseConfig"].value("maxThreads", vMaxThreads);
    vVerbose    = _j["baseConfig"].value("verbose", vVerbose);
    vInput      = _j["baseConfig"].value("input", vInput);
  }

  if (_j.count("commands") == 0 || !_j["commands"].is_array()) return;
  for (auto const &i : _j["commands"]) {
    string lName = i.at("id").get<string>();

    auto lCmd = commandFromString(lName);
    if (!lCmd) {
      lLogger->warn("Command '{}' not found", lName);
      continue;
    }

    lCmd->fromJSON(i.at("info"));
    vCommands.push_back(lCmd);
  }
}

json Config::toJSON() const {
  auto lJSON =
      json{{"baseConfig", {{"name", vName}, {"maxThreads", vMaxThreads}, {"verbose", vVerbose}, {"input", vInput}}},
           {"commands", json::array()}};

  for (auto &i : vCommands)
    lJSON["commands"].push_back(json{{"id", i->getName()}, {"info", i->toJSON()}});

  return lJSON;
}

bool Config::addCommand(std::string _name) noexcept {
  auto lLogger = getLogger();
  auto lCmd    = commandFromString(_name);
  if (!lCmd) {
    lLogger->error("Command '{}' not found", _name);
    return false;
  }

  vCommands.push_back(lCmd);
  return true;
}

std::vector<Config::CMD_PTR> Config::getCommands() { return vCommands; }
