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

#include "ImportMesh.hpp"

using namespace BVHTest::IO;
using namespace BVHTest::base;

ImportMesh::~ImportMesh() {}

void ImportMesh::fromJSON(const json &_j) {
  vSourcePath = _j.at("sourcePath").get<std::string>();
  vDestPath   = _j.at("destPath").get<std::string>();
}

json ImportMesh::toJSON() const { return json{{"sourcePath", vSourcePath}, {"destPath", vDestPath}}; }

ErrorCode ImportMesh::runImpl(State &_state) {
  (void)_state;
  return ErrorCode::OK;
}
