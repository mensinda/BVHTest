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
#include "minilzo-2.10/minilzo.h"
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

ErrorCode Load::loadVers1(State &_state, json &_cfg, fstream &_binFile) {
  auto lLogger = getLogger();
  lLogger->warn("File version 1 is deprecated. Consider converting to the new format");

  size_t lVert   = _cfg.at("mesh").at("vert").get<size_t>();
  size_t lNormal = _cfg.at("mesh").at("normals").get<size_t>();
  size_t lFaces  = _cfg.at("mesh").at("faces").get<size_t>();

  _state.mesh.vert.resize(lVert);
  _state.mesh.norm.resize(lNormal);
  _state.mesh.faces.resize(lFaces);

  _binFile.read(reinterpret_cast<char *>(_state.mesh.vert.data()), lVert * sizeof(glm::vec3));
  _binFile.read(reinterpret_cast<char *>(_state.mesh.norm.data()), lNormal * sizeof(glm::vec3));
  _binFile.read(reinterpret_cast<char *>(_state.mesh.faces.data()), lFaces * sizeof(Triangle));
  _binFile.close();

  return ErrorCode::OK;
}

ErrorCode Load::loadVers2(State &_state, json &_cfg, fstream &_binFile) {
  auto lLogger = getLogger();

  size_t   lVert         = _cfg.at("mesh").at("vert").get<size_t>();
  size_t   lNormal       = _cfg.at("mesh").at("normals").get<size_t>();
  size_t   lFaces        = _cfg.at("mesh").at("faces").get<size_t>();
  uint32_t lCheckSumComp = _cfg.at("compressedChecksum").get<uint32_t>();
  uint32_t lCheckSumRaw  = _cfg.at("rawChecksum").get<uint32_t>();

  // Read File
  auto lBeginPos = _binFile.tellg();
  _binFile.seekg(0, _binFile.end);
  size_t lCompSize = _binFile.tellg() - lBeginPos;
  size_t lDataSize = (lVert + lNormal) * sizeof(vec3) + lFaces * sizeof(Triangle);
  _binFile.seekg(0, _binFile.beg);

  unique_ptr<uint8_t[]> lComp = unique_ptr<uint8_t[]>(new uint8_t[lCompSize]);
  unique_ptr<uint8_t[]> lData = unique_ptr<uint8_t[]>(new uint8_t[lDataSize]);

  _binFile.read(reinterpret_cast<char *>(lComp.get()), lCompSize);
  _binFile.close();

  auto lCheckSumTemp = lzo_adler32(0, nullptr, 0);
  lCheckSumTemp      = lzo_adler32(lCheckSumTemp, lComp.get(), lCompSize);

  if (lCheckSumTemp != lCheckSumComp) {
    lLogger->error("Corrupt binary file");
    return ErrorCode::IO_ERROR;
  }

  auto lRet = lzo1x_decompress(lComp.get(), lCompSize, lData.get(), &lDataSize, nullptr);
  if (lRet != LZO_E_OK) {
    lLogger->error("Decompression of binary file failed: {}", lRet);
    return ErrorCode::IO_ERROR;
  }

  lComp = nullptr; // free memory

  lCheckSumTemp = lzo_adler32(0, nullptr, 0);
  lCheckSumTemp = lzo_adler32(lCheckSumTemp, lData.get(), lDataSize);

  if (lCheckSumTemp != lCheckSumRaw) {
    lLogger->error("Decompression of binary file failed: corrupt data");
    return ErrorCode::IO_ERROR;
  }

  _state.mesh.vert.resize(lVert);
  _state.mesh.norm.resize(lNormal);
  _state.mesh.faces.resize(lFaces);

  size_t lOffset = 0;
  memcpy(_state.mesh.vert.data(), lData.get() + lOffset, lVert * sizeof(vec3));
  lOffset += lVert * sizeof(vec3);
  memcpy(_state.mesh.norm.data(), lData.get() + lOffset, lNormal * sizeof(vec3));
  lOffset += lNormal * sizeof(vec3);
  memcpy(_state.mesh.faces.data(), lData.get() + lOffset, lFaces * sizeof(Triangle));

  return ErrorCode::OK;
}


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

  if (!fs::exists(lBinPath) || !fs::is_regular_file(lBinPath)) {
    lLogger->error("File {} does not exist", lBinPath.string());
    return ErrorCode::IO_ERROR;
  }

  fstream lBinFile(lBinPath.string(), lBinFile.in | lBinFile.binary);
  if (!lBinFile.is_open()) {
    lLogger->error("Failed to open file {}", lBinPath.string());
    return ErrorCode::IO_ERROR;
  }

  switch (lVers) {
    case 1: return loadVers1(_state, lCfg, lBinFile);
    case 2: return loadVers2(_state, lCfg, lBinFile);
    default:
      lLogger->error("File format version is {} but {} is required", lVers, vLoaderVersion);
      return ErrorCode::PARSE_ERROR;
  };

  return ErrorCode::OK;
}
