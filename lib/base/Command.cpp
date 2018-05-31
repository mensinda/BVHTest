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

#if __has_include(<sys/ioctl.h>) && __has_include(<unistd.h>)
#  define ENABLE_IOCTL
#  include <sys/ioctl.h>
#  include <unistd.h>
#endif

using namespace BVHTest::base;
using namespace BVHTest::Enum2Str;
using namespace std;
using namespace std::chrono;

Command::~Command() {}

ErrorCode Command::run(State &_state) {
  auto lLogger = getLogger();
  auto lName   = getName();
  lLogger->info("Running:  [{:^10}] command {:<16} -- {}", toStr(getType()), lName, getDesc());

  if ((_state.commandsRun & getRequiredCommands()) != getRequiredCommands()) {
    lLogger->error("Command {} has unmet requirements: Required: {}", lName);
    lLogger->error("  - Required:         {}", base_ErrorCode_toStr(getRequiredCommands()));
    lLogger->error("  - Already executed: {}", base_ErrorCode_toStr(_state.commandsRun));
  }

  auto      lStart = high_resolution_clock::now();
  ErrorCode lRet   = runImpl(_state);
  auto      lEnd   = high_resolution_clock::now();

  fmt::print("\x1b[2K\x1b[1G"); // Clear progress line

  if (lRet != ErrorCode::OK) {
    lLogger->error("Command {} returned {}", lName, toStr(lRet));
    return lRet;
  }

  _state.commandsRun |= static_cast<uint64_t>(getType());
  _state.commands.push_back({lName, duration_cast<milliseconds>(lEnd - lStart)});
  return ErrorCode::OK;
}

#define ESC_SEQ "\x1b[2K\x1b[1G\x1b[1m"
#define ESC_END "\x1b[0m"

void Command::progress(std::string _str, float _val) {
  _val = max(_val, 0.0f);
  _val = min(_val, 1.0f);

  uint32_t lWidth = 100;

#ifdef ENABLE_IOCTL
  struct winsize ws;
  if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) { lWidth = ws.ws_col; }
  lWidth = min(lWidth, 1000u);
#endif

  lWidth -= 58;
  uint32_t lPWidth = static_cast<uint32_t>(_val * lWidth);
  uint32_t lVal    = static_cast<uint32_t>(_val * 100);
  string   lFMT;
  if (lPWidth == 0) {
    lFMT = fmt::format("[{{0: >{}}}]", lWidth);
  } else if (lPWidth == lWidth) {
    lFMT = fmt::format("[{{0:#>{}}}]", lWidth);
  } else {
    lFMT = fmt::format("[{{0:#>{}}}{{1: >{}}}]", lPWidth, lWidth - lPWidth);
  }
  fmt::print("\x1b[2K\x1b[1G\x1b[1m{2:<50}" + lFMT + " {3:>3}%\x1b[0m", "", "", _str, lVal);
  fflush(stdout);
}

void Command::progress(std::string _str, uint32_t _curr, uint32_t _max) {
  progress(_str, static_cast<float>(_curr) / static_cast<float>(_max));
}
