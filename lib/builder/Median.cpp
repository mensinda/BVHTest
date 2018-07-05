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

#include "Median.hpp"
#include <algorithm>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;

Median::~Median() {}
void Median::fromJSON(const json &) {}
json Median::toJSON() const { return json::object(); }

//! \todo Complete this!
Median::ITER Median::split(Median::ITER _begin, Median::ITER _end, uint32_t) { return _begin + ((_end - _begin) / 2); }


ErrorCode Median::runImpl(State &_state) {
  _state.bvh.reserve(_state.mesh.faces.size() * 2); // Assuming perfect binary tree

  auto lAABBs = boundingVolumesFromMesh(_state.mesh);
  build(begin(lAABBs), end(lAABBs), _state.bvh);

  return ErrorCode::OK;
}
