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
#include "CameraExport.hpp"
#include "Enum2Str.hpp"
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

CameraExport::~CameraExport() {}
void CameraExport::fromJSON(const json &) {}
json CameraExport::toJSON() const { return json::object(); }

ErrorCode CameraExport::runImpl(State &_state) {
  auto lLogger = getLogger();

  fs::path lBasePath = _state.basePath;
  lBasePath          = lBasePath / _state.input;
  fs::path lOutDir;
  if (fs::is_directory(lBasePath)) {
    lOutDir = lBasePath;
  } else if (fs::is_regular_file(lBasePath)) {
    lOutDir = lBasePath.parent_path();
    if (!fs::is_directory(lOutDir)) {
      lLogger->error("Path {} is not a directory", fs::absolute(lOutDir).string());
      return ErrorCode::IO_ERROR;
    }
  } else {
    lLogger->error("Invalid base path {}", fs::absolute(lBasePath).string());
    return ErrorCode::IO_ERROR;
  }

  fs::path lControlPath = fs::absolute(lOutDir) / "cameras.json";

  vector<json> lCameras;
  for (auto const &i : _state.cameras) {
    lCameras.push_back(json{{"type", Enum2Str::toStr(i->getType())}, {"cam", i->toJSON()}});
  }

  json lControlData = {{"version", vFormatVers}, {"cameras", lCameras}};

  fstream lControlFile(lControlPath.string(), lControlFile.out | lControlFile.trunc);

  if (!lControlFile.is_open()) {
    lLogger->error("Failed to open {}.", lControlPath.string());
    return ErrorCode::IO_ERROR;
  }

  lControlFile << lControlData.dump(2);
  lControlFile.close();

  return ErrorCode::OK;
}
