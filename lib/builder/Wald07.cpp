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

#include "Wald07.hpp"

using namespace std;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;

Wald07::~Wald07() {}

void Wald07::fromJSON(const json &) {}
json Wald07::toJSON() const { return json::object(); }

ErrorCode Wald07::runImpl(State &_state) {
  _state.aabbs = boundingVolumesFromMesh(_state.mesh);
  return ErrorCode::OK;
}
