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

#define HEAP_ALLOC(var, size) lzo_align_t __LZO_MMODEL var[((size) + (sizeof(lzo_align_t) - 1)) / sizeof(lzo_align_t)]

using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::IO;

BVHExport::~BVHExport() {}

void BVHExport::fromJSON(const json &_j) { vExportName = _j.value("name", vExportName); }
json BVHExport::toJSON() const { return json{{"name", vExportName}}; }

ErrorCode BVHExport::runImpl(State &_state) {
  auto lLogger = getLogger();

  static HEAP_ALLOC(wrkmem, LZO1X_1_MEM_COMPRESS);

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


  // Compress
  size_t lNumNodes    = _state.bvh.size();
  size_t lElementSize = sizeof(AABB) + 4 * sizeof(uint32_t) + sizeof(uint16_t) + sizeof(uint8_t) + sizeof(float);
  size_t lInSize      = lNumNodes * lElementSize + _state.mesh.faces.size() * sizeof(Triangle);
  size_t lCompSize    = lInSize + (lInSize / 16) + 64 + 3;
  unique_ptr<uint8_t[]> lData = unique_ptr<uint8_t[]>(new uint8_t[lInSize]);
  unique_ptr<uint8_t[]> lComp = unique_ptr<uint8_t[]>(new uint8_t[lCompSize]);

  size_t  lOffset  = 0;
  BVHNode lBVHData = _state.bvh.data();

  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lBVHData.bbox), lNumNodes * sizeof(AABB));
  lOffset += lNumNodes * sizeof(AABB);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lBVHData.parent), lNumNodes * sizeof(uint32_t));
  lOffset += lNumNodes * sizeof(uint32_t);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lBVHData.numChildren), lNumNodes * sizeof(uint32_t));
  lOffset += lNumNodes * sizeof(uint32_t);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lBVHData.left), lNumNodes * sizeof(uint32_t));
  lOffset += lNumNodes * sizeof(uint32_t);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lBVHData.right), lNumNodes * sizeof(uint32_t));
  lOffset += lNumNodes * sizeof(uint32_t);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lBVHData.isLeft), lNumNodes * sizeof(uint8_t));
  lOffset += lNumNodes * sizeof(uint8_t);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lBVHData.level), lNumNodes * sizeof(uint16_t));
  lOffset += lNumNodes * sizeof(uint16_t);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lBVHData.surfaceArea), lNumNodes * sizeof(float));
  lOffset += lNumNodes * sizeof(float);
  memcpy(lData.get() + lOffset,
         reinterpret_cast<char *>(_state.mesh.faces.data()),
         _state.mesh.faces.size() * sizeof(Triangle));

  auto lCheckSumRaw = lzo_adler32(0, nullptr, 0);
  lCheckSumRaw      = lzo_adler32(lCheckSumRaw, lData.get(), lInSize);

  auto lRet = lzo1x_1_compress(lData.get(), lInSize, lComp.get(), &lCompSize, wrkmem);
  if (lRet != LZO_E_OK) {
    lLogger->error("Compression failed: {}", lRet);
    return ErrorCode::IO_ERROR;
  }

  auto lCheckSumComp = lzo_adler32(0, nullptr, 0);
  lCheckSumComp      = lzo_adler32(lCheckSumComp, lComp.get(), lCompSize);


  json lControlData = {{"version", vFormatVers},
                       {"bin", (vExportName + "_bvh.bin")},
                       {"BVHSize", _state.bvh.size()},
                       {"numTris", _state.mesh.faces.size()},
                       {"treeHeight", _state.bvh.maxLevel()},
                       {"compressedChecksum", lCheckSumComp},
                       {"rawChecksum", lCheckSumRaw}};

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

  lBinaryFile.write(reinterpret_cast<char *>(lComp.get()), lCompSize);
  lBinaryFile.close();

  return ErrorCode::OK;
}
