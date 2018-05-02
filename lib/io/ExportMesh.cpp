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

using namespace BVHTest::IO;
using namespace BVHTest::base;

ExportMesh::~ExportMesh() {}
void ExportMesh::fromJSON(const json &_j) { vOutDir = _j.value("outDir", vOutDir); }
json ExportMesh::toJSON() const { return json{{"outDir", vOutDir}}; }

ErrorCode ExportMesh::runImpl(State &_state) {
  (void)_state;
  auto     lLogger = getLogger();
  fs::path outDir  = fs::absolute(vOutDir);
  if (!fs::exists(outDir)) { fs::create_directory(outDir); }

  if (!fs::is_directory(outDir)) {
    lLogger->error("Path {} is not a directory", outDir.string());
    return ErrorCode::IO_ERROR;
  }

  lLogger->info("Output dir: {}", outDir.string());

  return ErrorCode::OK;
}
