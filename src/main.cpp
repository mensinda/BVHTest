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
#include "base/Config.hpp"
#include "base/StatusDump.hpp"
#include "base/StringHash.hpp"
#include "builder/Median.hpp"
#include "builder/Wald07.hpp"
#include "io/BVHExport.hpp"
#include "io/BVHImport.hpp"
#include "io/CameraExport.hpp"
#include "io/CameraImport.hpp"
#include "io/ExportMesh.hpp"
#include "io/ImportMesh.hpp"
#include "io/Load.hpp"
#include "tracer/CPUTracer.hpp"
#include "viewer/Viewer.hpp"
#include "Enum2Str.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace fmt;
using namespace Enum2Str;
using namespace nlohmann;

const vector<string> gCommandList = {"import",
                                     "load",
                                     "export",
                                     "BVHExport",
                                     "BVHImport",
                                     "camExport",
                                     "camImport",
                                     "median",
                                     "wald07",
                                     "CPUTracer",
                                     "status",
                                     "viewer"};

// String command to object
Config::CMD_PTR fromString(string _s) {
  switch (fnv1aHash(_s)) {
    case "import"_h: return make_shared<IO::ImportMesh>();
    case "load"_h: return make_shared<IO::Load>();
    case "export"_h: return make_shared<IO::ExportMesh>();
    case "BVHExport"_h: return make_shared<IO::BVHExport>();
    case "BVHImport"_h: return make_shared<IO::BVHImport>();
    case "camExport"_h: return make_shared<IO::CameraExport>();
    case "camImport"_h: return make_shared<IO::CameraImport>();
    case "median"_h: return make_shared<builder::Median>();
    case "wald07"_h: return make_shared<builder::Wald07>();
    case "CPUTracer"_h: return make_shared<tracer::CPUTracer>();
    case "status"_h: return make_shared<base::StatusDump>();
    case "viewer"_h: return make_shared<view::Viewer>();
  }
  return nullptr;
}

int usage() {
  cout << "Options:" << endl
       << "  help              -- display this help message" << endl
       << "  run CFG           -- run BVHTest with configuration file CFG" << endl
       << "  list              -- list all commands and exit" << endl
       << "  genrate FILE CMDs -- generate config file with commands CMDs (use - for stdout)" << endl
       << endl;

  return 0;
}

int list() {
  auto lLogger = getLogger();
  cout << "Commands:" << endl;
  for (string const &i : gCommandList) {
    auto lCmd = fromString(i);
    if (!lCmd) {
      lLogger->error("Internal error: fromString returned nullptr for {}", i);
      return 1;
    }

    cout << "  - {:<10} [{:^10}] -- {}"_format(i, toStr(lCmd->getType()), lCmd->getDesc()) << endl;
  }
  return 0;
}

int generate(vector<string> &args, size_t _start = 1) {
  auto lLogger = getLogger();
  if (args.size() < _start + 1) {
    lLogger->error("to view arguments for generate");
    return 1;
  }

  Config lCfg([](string _s) { return fromString(_s); });

  for (size_t i = _start + 1; i < args.size(); ++i) {
    if (!lCfg.addCommand(args[i])) { return 2; }
  }

  json lOutCfg = lCfg;
  if (args[_start] == "-") {
    cout << lOutCfg.dump(2);
  } else {
    fstream lFile(args[_start], lFile.out);
    if (!lFile.is_open()) {
      lLogger->error("Failed to open {} for writing", args[_start]);
      return 1;
    }

    lLogger->info("Writing file {}", args[_start]);
    lFile << lOutCfg.dump(2);
  }

  return 0;
}

bool run(string _file) {
  auto lLogger = getLogger();

  try {
    // Parse and update config
    fstream lFile(_file, lFile.in);
    json    lJSON;
    lFile >> lJSON;
    lFile.close();

    Config lCfg([](string _s) { return fromString(_s); });
    lCfg.fromJSON(lJSON);

    lJSON = lCfg;
    lFile.open(_file, lFile.out | lFile.trunc);
    lFile << lJSON.dump(2);
    lFile.close();

    // Start executing
    if (lCfg.getIsVerbose()) {
      lLogger->set_level(spd::level::debug);
    } else {
      lLogger->set_level(spd::level::warn);
    }

    auto lInput    = lCfg.getInput();
    auto lCommands = lCfg.getCommands();

    for (auto &i : lInput) {
      lLogger->info("Processing {}", i);
      State lState;
      lState.input    = i;
      lState.basePath = lCfg.getBasePath();

      for (auto &j : lCommands)
        if (j->run(lState) != ErrorCode::OK) break;
    }

  } catch (detail::exception &e) {
    lLogger->error("JSON exception: {}", e.what());
    return false;
  }

  return true;
}

int main(int argc, char *argv[]) {
  auto           lLogger = getLogger();
  vector<string> args;
  for (int i = 1; i < argc; ++i)
    args.push_back(argv[i]);

  if (args.size() == 0) return usage();

  for (size_t i = 0; i < args.size(); ++i) {
    switch (fnv1aHash(args[i])) {
      case "list"_h: return list();
      case "generate"_h: return generate(args, i + 1);
      case "run"_h:
        if (++i >= args.size()) {
          lLogger->error("Argument required for 'run'");
          return 1;
        }
        if (!run(args[i])) return 2;
        break;

      case "help"_h:
      case "-h"_h:
      case "--help"_h:
      default: return usage();
    }
  }

  return 0;
}
