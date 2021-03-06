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
#include "BVHImport.hpp"
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

BVHImport::~BVHImport() {}
void BVHImport::fromJSON(const json &_j) { vExportName = _j.value("name", vExportName); }
json BVHImport::toJSON() const { return json{{"name", vExportName}}; }

ErrorCode BVHImport::runImpl(State &_state) {
  auto lLogger = getLogger();

  fs::path lBasePath = _state.basePath;
  lBasePath          = lBasePath / _state.input;
  fs::path lDataDir;
  if (fs::is_directory(lBasePath)) {
    lDataDir = lBasePath;
  } else if (fs::is_regular_file(lBasePath)) {
    lDataDir = lBasePath.parent_path();
    if (!fs::is_directory(lDataDir)) {
      lLogger->error("Path {} is not a directory", fs::absolute(lDataDir).string());
      return ErrorCode::IO_ERROR;
    }
  } else {
    lLogger->error("Invalid base path {}", fs::absolute(lBasePath).string());
    return ErrorCode::IO_ERROR;
  }

  fs::path lControlPath = fs::absolute(lDataDir) / (vExportName + "_bvh.json");
  if (!fs::exists(lControlPath) || !fs::is_regular_file(lControlPath)) {
    lLogger->error("Invalid BVH control file {}", lControlPath.string());
    return ErrorCode::IO_ERROR;
  }

  fstream lControlFile(lControlPath.string(), lControlFile.in);
  if (!lControlFile.is_open()) {
    lLogger->error("Failed to open BVH control file {} for reading", lControlPath.string());
    return ErrorCode::IO_ERROR;
  }

  json lControlData;
  lControlFile >> lControlData;
  lControlFile.close();

  uint32_t lVersion = lControlData.at("version").get<uint32_t>();

  if (lVersion != vFormatVers) {
    lLogger->error("Incompatible version {}! Version {} required", lVersion, vFormatVers);
    return ErrorCode::IO_ERROR;
  }

  fs::path lBinaryPath = lDataDir / lControlData.at("bin").get<string>();
  uint32_t lSize       = lControlData.at("BVHSize").get<uint32_t>();
  _state.bvh.setMaxLevel(lControlData.at("treeHeight").get<uint16_t>());
  //   uint32_t lCheckSumComp = lControlData.at("compressedChecksum").get<uint32_t>();
  uint32_t lCheckSumRaw = lControlData.at("rawChecksum").get<uint32_t>();

  if (!fs::exists(lBinaryPath) || !fs::is_regular_file(lBinaryPath)) {
    lLogger->error("Invalid BVH binary file {}", fs::absolute(lBinaryPath).string());
    return ErrorCode::IO_ERROR;
  }

  fstream lBinaryFile(lBinaryPath.string(), lBinaryFile.in | lBinaryFile.binary);
  if (!lBinaryFile.is_open()) {
    lLogger->error("Failed to open BVH binary file {} for reading", fs::absolute(lBinaryPath).string());
    return ErrorCode::IO_ERROR;
  }

  // Read File
  //   auto lBeginPos = lBinaryFile.tellg();
  lBinaryFile.seekg(0, lBinaryFile.end);
  //   size_t lCompSize = lBinaryFile.tellg() - lBeginPos;
  size_t lDataSize = lSize * sizeof(BVHNode);
  lBinaryFile.seekg(0, lBinaryFile.beg);

  //   unique_ptr<uint8_t[]> lComp = unique_ptr<uint8_t[]>(new uint8_t[lCompSize]);
  //   unique_ptr<uint8_t[]> lData = unique_ptr<uint8_t[]>(new uint8_t[lDataSize]);

  _state.bvh.resize(lSize);

  lBinaryFile.read(reinterpret_cast<char *>(_state.bvh.data()), lDataSize);
  lBinaryFile.close();

  //   auto lCheckSumTemp = lzo_adler32(0, nullptr, 0);
  //   lCheckSumTemp      = lzo_adler32(lCheckSumTemp, lComp.get(), lCompSize);

  //   if (lCheckSumTemp != lCheckSumComp) {
  //     lLogger->error("Corrupt binary file '{}'", lBinaryPath.string());
  //     return ErrorCode::IO_ERROR;
  //   }
  //
  //   auto lRet = lzo1x_decompress(lComp.get(), lCompSize, lData.get(), &lDataSize, nullptr);
  //   if (lRet != LZO_E_OK) {
  //     lLogger->error("Decompression of '{}' failed: {}", lBinaryPath.string(), lRet);
  //     return ErrorCode::IO_ERROR;
  //   }
  //
  //   lComp = nullptr; // Free memory

  auto lCheckSumTemp = lzo_adler32(0, nullptr, 0);
  lCheckSumTemp      = lzo_adler32(lCheckSumTemp, reinterpret_cast<uint8_t *>(_state.bvh.data()), lDataSize);

  if (lCheckSumTemp != lCheckSumRaw) {
    lLogger->error("Decompression of '{}' failed: corrupt data", lBinaryPath.string());
    return ErrorCode::IO_ERROR;
  }

  return ErrorCode::OK;
}
