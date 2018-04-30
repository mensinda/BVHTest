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
#include "base/StringHash.hpp"
#include "io/ImportMesh.hpp"
#include "Enum2Str.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::IO;
using namespace fmt;
using namespace Enum2Str;

const vector<string> gCommandList = {"import"};

// String command to object
Config::CMD_PTR fromString(string _s) {
  switch (fnv1aHash(_s)) {
    case "import"_h: return make_shared<ImportMesh>();
  }
  return nullptr;
}

int usage() {
  cout << "Options:" << endl
       << "  help              -- display this help message" << endl
       << "  run CFG           -- run BVHTest with configuration file CFG" << endl
       << "  list              -- list all commands and exit" << endl
       << "  genrate FILE CMDs -- generate config file with commands CMDs" << endl
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
  json   lOutCfg = lCfg;

  fstream lFile(args[_start], lFile.out);
  if (!lFile.is_open()) {
    lLogger->error("Failed to open {} for writing", args[_start]);
    return 1;
  }

  lLogger->info("Writing file {}", args[_start]);
  lFile << lOutCfg.dump(2) << endl;

  return 0;
}

int main(int argc, char *argv[]) {
  vector<string> args;
  for (int i = 1; i < argc; ++i)
    args.push_back(argv[i]);

  if (args.size() == 0)
    return usage();

  switch (fnv1aHash(args[0])) {
    case "list"_h: return list();
    case "generate"_h: return generate(args);
    case "help"_h:
    case "-h"_h:
    case "--help"_h:
    default: return usage();
  }

  return 0;
}
