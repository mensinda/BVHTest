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
#include "Command.hpp"
#include "Enum2Str.hpp"
#include <chrono>

using namespace BVHTest::base;
using namespace BVHTest::Enum2Str;
using namespace std;
using namespace std::chrono;

Command::~Command() {}

ErrorCode Command::run(State &_state) {
  auto lLogger = getLogger();
  lLogger->info("Running:  [{:^10}] command {:<10} -- {}", toStr(getType()), getName(), getDesc());

  if ((_state.commandsRun & getRequiredCommands()) != getRequiredCommands()) {
    lLogger->error("Command {} has unmet requirements: Required: {}", getName());
    lLogger->error("  - Required:         {}", base_ErrorCode_toStr(getRequiredCommands()));
    lLogger->error("  - Already executed: {}", base_ErrorCode_toStr(_state.commandsRun));
  }

  auto      lStart = high_resolution_clock::now();
  ErrorCode lRet   = runImpl(_state);
  auto      lEnd   = high_resolution_clock::now();

  if (lRet != ErrorCode::OK) {
    lLogger->error("Command {} returned {}", getName(), toStr(lRet));
    return lRet;
  }

  milliseconds lDur = duration_cast<milliseconds>(lEnd - lStart);
  auto         lSec = duration_cast<seconds>(lDur).count();
  lLogger->info("Duration: {:>3}s {:>3}ms", lSec, lDur.count() - lSec * 1000);

  _state.commandsRun |= static_cast<uint64_t>(getType());
  _state.commands.push_back({getName(), static_cast<uint64_t>(lDur.count())});
  return ErrorCode::OK;
}
