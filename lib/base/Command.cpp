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

  ErrorCode lRet = setup(_state);
  if (lRet != ErrorCode::OK) {
    lLogger->error("Setup function of command {} returned {}", lName, toStr(lRet));
    return lRet;
  }

  auto lStart = high_resolution_clock::now();
  lRet        = runImpl(_state);
  auto lEnd   = high_resolution_clock::now();

  PROGRESS_DONE;

  teardown(_state);

  switch (lRet) {
    case ErrorCode::OK: break;
    case ErrorCode::WARNING: lLogger->warn("Command {} returned a warning", lName); break;
    default: lLogger->error("Command {} returned {}", lName, toStr(lRet)); return lRet;
  }

  _state.commandsRun |= static_cast<uint64_t>(getType());
  _state.commands.push_back({lName, getType(), duration_cast<milliseconds>(lEnd - lStart)});
  return ErrorCode::OK;
}

void Command::progressDone() {
  if (isatty(fileno(stdout))) { fmt::print("\x1b[2K\x1b[1G"); } // Clear progress line
}


#define ESC_SEQ "\x1b[2K\x1b[1G\x1b[1m"
#define ESC_END "\x1b[0m"

void Command::progress(std::string _str, float _val) {
  _val = max(_val, 0.0f);
  _val = min(_val, 1.0f);

  uint32_t lWidth  = 100;
  string   lSuffix = "\x1b[0m";
  string   lPrefix = "\x1b[2K\x1b[1G\x1b[1m";

#ifdef ENABLE_IOCTL
  struct winsize ws;
  if (isatty(fileno(stdout))) {
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0) { lWidth = ws.ws_col; }
    lWidth = min(lWidth, 1000u);
  } else {
    lSuffix = "\n";
    lPrefix = "";
  }
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
  fmt::print("{4}{2:<50}" + lFMT + " {3:>3}%{5}", "", "", _str, lVal, lPrefix, lSuffix);
  fflush(stdout);
}

void Command::progress(std::string _str, uint32_t _curr, uint32_t _max) {
  progress(_str, static_cast<float>(_curr) / static_cast<float>(_max));
}
