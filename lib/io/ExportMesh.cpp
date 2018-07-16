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
#include "ExportMesh.hpp"
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
using namespace BVHTest::IO;
using namespace BVHTest::base;

ExportMesh::~ExportMesh() {}
void ExportMesh::fromJSON(const json &_j) { vOutDir = _j.value("outDir", vOutDir); }
json ExportMesh::toJSON() const { return json{{"outDir", vOutDir}}; }

bool checkDir(fs::path _p) {
  auto lLogger = getLogger();
  if (!fs::exists(_p)) { fs::create_directory(_p); }
  if (!fs::is_directory(_p)) {
    lLogger->error("Path {} is not a directory", _p.string());
    return false;
  }
  return true;
}

ErrorCode ExportMesh::runImpl(State &_state) {
  auto     lLogger = getLogger();
  fs::path outDir  = fs::absolute(vOutDir);
  fs::path dataDir = outDir / fs::absolute(_state.input).filename().replace_extension("");

  //   static HEAP_ALLOC(wrkmem, LZO1X_1_MEM_COMPRESS);

  if (!checkDir(outDir)) { return ErrorCode::IO_ERROR; }
  if (!checkDir(dataDir)) { return ErrorCode::IO_ERROR; }

  outDir  = fs::canonical(outDir);
  dataDir = fs::canonical(dataDir);

  auto &lMesh = _state.mesh;

  // Compress
  size_t lInSize = (lMesh.vert.size() + lMesh.norm.size()) * sizeof(vec3) + lMesh.faces.size() * sizeof(Triangle);
  //   size_t lCompSize = lInSize + (lInSize / 16) + 64 + 3;
  unique_ptr<uint8_t[]> lData = unique_ptr<uint8_t[]>(new uint8_t[lInSize]);
  //   unique_ptr<uint8_t[]> lComp = unique_ptr<uint8_t[]>(new uint8_t[lCompSize]);

  size_t lOffset = 0;
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lMesh.vert.data()), lMesh.vert.size() * sizeof(vec3));
  lOffset += lMesh.vert.size() * sizeof(vec3);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lMesh.norm.data()), lMesh.norm.size() * sizeof(vec3));
  lOffset += lMesh.norm.size() * sizeof(vec3);
  memcpy(lData.get() + lOffset, reinterpret_cast<char *>(lMesh.faces.data()), lMesh.faces.size() * sizeof(Triangle));

  auto lCheckSumRaw = lzo_adler32(0, nullptr, 0);
  lCheckSumRaw      = lzo_adler32(lCheckSumRaw, lData.get(), lInSize);

  //   auto lRet = lzo1x_1_compress(lData.get(), lInSize, lComp.get(), &lCompSize, wrkmem);
  //   if (lRet != LZO_E_OK) {
  //     lLogger->error("Compression failed: {}", lRet);
  //     return ErrorCode::IO_ERROR;
  //   }

  //   auto lCheckSumComp = lzo_adler32(0, nullptr, 0);
  //   lCheckSumComp      = lzo_adler32(lCheckSumComp, lComp.get(), lCompSize);

  fs::path lControl = dataDir / "data.json";
  fs::path lBinary  = dataDir / "data.bin";

  json lCfg = {
      {"version", vFormatVers},
      {"bin", lBinary.filename().string()},
      //       {"compressedChecksum", lCheckSumComp},
      {"rawChecksum", lCheckSumRaw},
      {"mesh",
       {{"vert", _state.mesh.vert.size()}, {"normals", _state.mesh.norm.size()}, {"faces", _state.mesh.faces.size()}}}};

  fstream lControlFile(lControl.string(), lControlFile.out | lControlFile.trunc);
  fstream lBin(lBinary.string(), lBin.out | lBin.trunc | lBin.binary);

  if (!lControlFile.is_open()) {
    lLogger->error("Failed to open {}", lControl.string());
    return ErrorCode::IO_ERROR;
  }

  if (!lBin.is_open()) {
    lLogger->error("Failed to open {}", lBinary.string());
    return ErrorCode::IO_ERROR;
  }

  lControlFile << lCfg.dump(2);
  lControlFile.close();

  lBin.write(reinterpret_cast<char *>(lData.get()), lInSize);
  lBin.close();

  return ErrorCode::OK;
}
