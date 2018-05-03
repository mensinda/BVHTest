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
#include <fstream>

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
  (void)_state;
  auto     lLogger = getLogger();
  fs::path outDir  = fs::absolute(vOutDir);
  fs::path dataDir = outDir / fs::absolute(_state.input).filename().replace_extension("");

  if (!checkDir(outDir)) { return ErrorCode::IO_ERROR; }
  if (!checkDir(dataDir)) { return ErrorCode::IO_ERROR; }

  outDir  = fs::canonical(outDir);
  dataDir = fs::canonical(dataDir);

  fs::path lControl = dataDir / "data.json";
  fs::path lBinary  = dataDir / "data.bin";

  json lCfg = {
      {"version", vFormatVers},
      {"bin", lBinary.filename().string()},
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

  lBin.write(reinterpret_cast<char *>(_state.mesh.vert.data()), _state.mesh.vert.size() * sizeof(Vertex));
  lBin.write(reinterpret_cast<char *>(_state.mesh.norm.data()), _state.mesh.norm.size() * sizeof(Vertex));
  lBin.write(reinterpret_cast<char *>(_state.mesh.faces.data()), _state.mesh.faces.size() * sizeof(Triangle));
  lBin.close();

  return ErrorCode::OK;
}
