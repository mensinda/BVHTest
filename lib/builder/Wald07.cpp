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
#include <algorithm>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;

Wald07::~Wald07() {}

void Wald07::fromJSON(const json &) {}
json Wald07::toJSON() const { return json::object(); }

enum class Axis { NONE = -1, X = 0, Y = 1, Z = 2 };

/*!
 * \brief Partition sweep algorithm.
 *
 * Source: Wald et al. 2007 "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies"
 */
Wald07::PartitonRes Wald07::partitonSweep(ITER _begin, ITER _end) {
  if (_begin == _end) return {true, _end}; // ERROR
  if ((_end - _begin) == 1) return {true, _begin};

  float    lBestCost = getCostTri() * (_end - _begin);
  Axis     lBestAxis = Axis::NONE;
  uint32_t lBestEven = 0;

  for (Axis i : {Axis::X, Axis::Y, Axis::Z}) {
    sort(_begin, _end, [=](TriWithBB const &a, TriWithBB const &b) {
      switch (i) {
        case Axis::X: return a.centroid.x > b.centroid.x;
        case Axis::Y: return a.centroid.y > b.centroid.y;
        case Axis::Z: return a.centroid.z > b.centroid.z;
        default: return true;
      }
    });
  }

  return {false, _end};
}


ErrorCode Wald07::runImpl(State &_state) {
  _state.aabbs = boundingVolumesFromMesh(_state.mesh);
  partitonSweep(_state.aabbs.begin(), _state.aabbs.end());


  return ErrorCode::OK;
}
