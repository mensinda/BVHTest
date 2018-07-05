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
#include "CameraImport.hpp"
#include "misc/Camera.hpp"
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
using namespace BVHTest::misc;
using namespace BVHTest::base;
using namespace BVHTest::IO;

CameraImport::~CameraImport() {}
void CameraImport::fromJSON(const json &_j) { vAbortOnError = _j.value("abortOnError", vAbortOnError); }
json CameraImport::toJSON() const { return json{{"abortOnError", vAbortOnError}}; }

ErrorCode CameraImport::runImpl(State &_state) {
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
      return vAbortOnError ? ErrorCode::IO_ERROR : ErrorCode::OK;
    }
  } else {
    lLogger->error("Invalid base path {}", fs::absolute(lBasePath).string());
    return vAbortOnError ? ErrorCode::IO_ERROR : ErrorCode::OK;
  }

  fs::path lControlPath = fs::absolute(lOutDir) / "cameras.json";
  if (!fs::exists(lControlPath) || !fs::is_regular_file(lControlPath)) {
    lLogger->error("Path '{}' is not a regular file", lControlPath.string());
    return vAbortOnError ? ErrorCode::IO_ERROR : ErrorCode::OK;
  }

  fstream lControlFile(lControlPath.string(), lControlFile.in);
  if (!lControlFile.is_open()) {
    lLogger->error("Failed to open camera file '{}'", lControlPath.string());
    return vAbortOnError ? ErrorCode::IO_ERROR : ErrorCode::OK;
  }

  json lData;
  lControlFile >> lData;
  lControlFile.close();

  uint32_t lVersion = lData.at("version").get<uint32_t>();

  if (lVersion != vFormatVers) {
    lLogger->error("Incompatible version {}! Version {} required", lVersion, vFormatVers);
    return vAbortOnError ? ErrorCode::IO_ERROR : ErrorCode::OK;
  }

  std::vector<json> lCameras = lData.at("cameras").get<vector<json>>();
  for (json i : lCameras) {
    string lType = i.at("type").get<string>();
    if (lType != "PERSPECTIVE") {
      lLogger->warn("Currently only PERSPECTIVE type cameras supported. ({} provided)", lType);
      continue;
    }

    json lCam    = i.at("cam");
    auto lCamPtr = make_shared<Camera>();
    lCamPtr->fromJSON(lCam);
    _state.cameras.push_back(lCamPtr);
  }

  if (lData.count("camTrac") > 0) {
    std::vector<json> lCamTrac = lData.at("camTrac").get<vector<json>>();
    for (json i : lCamTrac) {
      auto lCamPtr = make_shared<Camera>();
      lCamPtr->fromJSON(i);
      _state.camTrac.push_back(lCamPtr);
    }
  }

  return ErrorCode::OK;
}
