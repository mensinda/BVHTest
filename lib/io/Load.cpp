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
#include "Load.hpp"
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

Load::~Load() {}
void Load::fromJSON(const json &) {}
json Load::toJSON() const { return json::object(); }

ErrorCode Load::runImpl(State &_state) {
  auto lLogger = getLogger();

  fs::path lPath = fs::absolute(_state.basePath) / _state.input;
  if (fs::is_directory(lPath)) lPath /= "data.json";

  if (!fs::exists(lPath) || !fs::is_regular_file(lPath)) {
    lLogger->error("File {} does not exist", lPath.string());
    return ErrorCode::IO_ERROR;
  }

  fstream lControl(lPath.string(), lControl.in);
  if (!lControl.is_open()) {
    lLogger->error("Failed to open file {}", lPath.string());
    return ErrorCode::IO_ERROR;
  }

  json lCfg;
  lControl >> lCfg;
  lControl.close();

  uint32_t lVers    = lCfg.at("version").get<uint32_t>();
  fs::path lBinPath = lPath.parent_path() / lCfg.at("bin").get<string>();
  size_t   lVert    = lCfg.at("mesh").at("vert").get<size_t>();
  size_t   lNormal  = lCfg.at("mesh").at("normals").get<size_t>();
  size_t   lFaces   = lCfg.at("mesh").at("faces").get<size_t>();

  if (lVers != vLoaderVersion) {
    lLogger->error("File format version is {} but {} is required", lVers, vLoaderVersion);
    return ErrorCode::PARSE_ERROR;
  }

  if (!fs::exists(lBinPath) || !fs::is_regular_file(lBinPath)) {
    lLogger->error("File {} does not exist", lBinPath.string());
    return ErrorCode::IO_ERROR;
  }

  fstream lBinFile(lBinPath.string(), lBinFile.in | lBinFile.binary);
  if (!lBinFile.is_open()) {
    lLogger->error("Failed to open file {}", lBinPath.string());
    return ErrorCode::IO_ERROR;
  }

  _state.mesh.vert.resize(lVert);
  _state.mesh.norm.resize(lNormal);
  _state.mesh.faces.resize(lFaces);

  lBinFile.read(reinterpret_cast<char *>(_state.mesh.vert.data()), lVert * sizeof(glm::vec3));
  lBinFile.read(reinterpret_cast<char *>(_state.mesh.norm.data()), lNormal * sizeof(glm::vec3));
  lBinFile.read(reinterpret_cast<char *>(_state.mesh.faces.data()), lFaces * sizeof(Triangle));
  lBinFile.close();

  return ErrorCode::OK;
}
