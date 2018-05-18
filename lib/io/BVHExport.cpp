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
#include "BVHExport.hpp"
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

BVHExport::~BVHExport() {}

void BVHExport::fromJSON(const json &_j) { vExportName = _j.value("name", vExportName); }
json BVHExport::toJSON() const { return json{{"name", vExportName}}; }

ErrorCode BVHExport::runImpl(State &_state) {
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

  fs::path lControlPath = fs::absolute(lOutDir) / (vExportName + "_bvh.json");
  fs::path lBinaryPath  = fs::absolute(lOutDir) / (vExportName + "_bvh.bin");

  json lControlData = {{"version", vFormatVers},
                       {"bin", (vExportName + "_bvh.bin")},
                       {"BVHSize", _state.bvh.size()},
                       {"numTris", _state.mesh.faces.size()},
                       {"treeHeight", _state.bvhMaxLevel}};

  fstream lControlFile(lControlPath.string(), lControlFile.out | lControlFile.trunc);
  fstream lBinaryFile(lBinaryPath.string(), lControlFile.out | lControlFile.trunc);

  if (!lControlFile.is_open()) {
    lLogger->error("Failed to open {}.", lControlPath.string());
    return ErrorCode::IO_ERROR;
  }

  if (!lBinaryFile.is_open()) {
    lLogger->error("Failed to open {}.", lControlPath.string());
    return ErrorCode::IO_ERROR;
  }

  lControlFile << lControlData.dump(2);
  lControlFile.close();

  lBinaryFile.write(reinterpret_cast<char *>(_state.bvh.data()), _state.bvh.size() * sizeof(BVH));
  lBinaryFile.write(reinterpret_cast<char *>(_state.mesh.faces.data()), _state.mesh.faces.size() * sizeof(Triangle));
  lBinaryFile.close();

  return ErrorCode::OK;
}
