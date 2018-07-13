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

#include "WriteData.hpp"
#include <fstream>

#if __has_include(<filesystem>)
#  include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#  include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#  error "std filesystem is not supported"
#endif

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::IO;

WriteData::~WriteData() {}
void WriteData::fromJSON(const json &_j) { (void)_j; }
json WriteData::toJSON() const { return json::object(); }

json WriteData::generateJSON(State &_state) {
  milliseconds lTotalTime = 0ms;

  for (auto const &i : _state.commands) {
    if (i.type == CommandType::BVH_OPT1 || i.type == CommandType::BVH_OPT2) { lTotalTime = i.duration; }
  }

  vector<json> lData;
  for (auto const &i : _state.optData.optStepps) {
    lData.push_back(json{{"step", i.step}, {"sah", i.sah}, {"duration", i.duration.count()}});
  }

  return json{{"totalTime", lTotalTime.count()},
              {"optStepps", lData},
              {"numSkipped", _state.optData.numSkipped},
              {"numTotal", _state.optData.numTotal}};
}


ErrorCode WriteData::runImpl(State &_state) {
  fs::path lBasePath = _state.basePath;
  lBasePath          = lBasePath / _state.input;
  fs::path lOutDir;
  if (fs::is_directory(lBasePath)) {
    lOutDir = lBasePath;
  } else if (fs::is_regular_file(lBasePath)) {
    lOutDir = lBasePath.parent_path();
    if (!fs::is_directory(lOutDir)) { return ErrorCode::IO_ERROR; }
  } else {
    return ErrorCode::IO_ERROR;
  }

  fs::path lOutPath = fs::absolute(lOutDir) / ("rData_"s + _state.name + ".json"s);
  fstream  lOutFile(lOutPath.string(), lOutFile.out | lOutFile.trunc);

  if (!lOutFile.is_open()) { return ErrorCode::IO_ERROR; }

  json lData = generateJSON(_state);
  lOutFile << lData.dump(2);
  lOutFile.close();
  return ErrorCode::OK;
}
