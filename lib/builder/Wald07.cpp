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
#include <thread>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;

Wald07::~Wald07() {}

enum class Axis { NONE = -1, X = 0, Y = 1, Z = 2 };

/*!
 * \brief Partition sweep algorithm.
 *
 * Source: Wald et al. 2007 "Ray Tracing Deformable Scenes Using Dynamic Bounding Volume Hierarchies"
 */
BuilderBase::ITER Wald07::split(ITER _begin, ITER _end, uint32_t) {
  size_t lSize = _end - _begin;
  if (lSize <= 2) { throw runtime_error("WALD07: Should not happen! " + to_string(__LINE__)); }

  float    lBestCost = numeric_limits<float>::infinity(); // getCostTri() * lSize;
  float    lThisCost = 0;
  float    lArea     = 0;
  Axis     lBestAxis = Axis::NONE;
  uint32_t lBestEven = 0;

  vector<float> lLeftArea;

  vector<TYPE> lXSort;
  vector<TYPE> lYSort;
  vector<TYPE> lZSort;

  lLeftArea.resize(lSize);

  lXSort.resize(lSize);
  lYSort.resize(lSize);
  lZSort.resize(lSize);

  AABB lTemp = _begin[0].bbox;
  for (size_t i = 0; i < lSize; ++i) {
    lTemp.mergeWith(_begin[i].bbox);
    lXSort[i] = _begin[i];
    lYSort[i] = _begin[i];
    lZSort[i] = _begin[i];
  }
  lArea = lTemp.surfaceArea();

  sort(begin(lXSort), end(lXSort), [](TCREF a, TCREF b) { return a.centroid.x > b.centroid.x; });
  sort(begin(lYSort), end(lYSort), [](TCREF a, TCREF b) { return a.centroid.y > b.centroid.y; });
  sort(begin(lZSort), end(lZSort), [](TCREF a, TCREF b) { return a.centroid.z > b.centroid.z; });

  for (Axis i : {Axis::X, Axis::Y, Axis::Z}) {
    vector<TYPE> &lRef = [&]() -> vector<TYPE> & {
      switch (i) {
        case Axis::X: return lXSort;
        case Axis::Y: return lYSort;
        case Axis::Z: return lZSort;
        default: throw runtime_error("WALD07: WTF just happened -- FIX THIS LOOP " + to_string(__LINE__));
      }
    }();

    AABB lLeft = lRef[0].bbox;
    for (size_t j = 1; j < lSize; ++j) {
      lLeftArea[j] = lLeft.surfaceArea();
      lLeft.mergeWith(lRef[j].bbox);
    }

    float lRightArea = 0;
    AABB  lRight     = lRef[lSize - 1].bbox;
    for (long int j = lSize - 2; j >= 1; --j) {
      lRightArea = lRight.surfaceArea();
      lRight.mergeWith(lRef[j].bbox);

      lThisCost = getCostInner() +                                         // Cost to intercect child boxes
                  ((lLeftArea[j] / lArea) * (j + 1) * getCostTri()) +      // Cost for the left side
                  ((lRightArea / lArea) * (lSize - j - 1) * getCostTri()); // Cost for the right side

      if (lThisCost < lBestCost) {
        lBestCost = lThisCost;
        lBestAxis = i;
        lBestEven = j;
      }
    }
  }

  if (lBestAxis == Axis::NONE) { throw runtime_error("WALD07: No axis found... " + to_string(__LINE__)); }

  // Copy back;
  vector<TYPE> &lBestSorted = [&]() -> vector<TYPE> & {
    switch (lBestAxis) {
      case Axis::X: return lXSort;
      case Axis::Y: return lYSort;
      case Axis::Z: return lZSort;
      default: throw runtime_error("WALD07: Thats not how enums work... " + to_string(__LINE__));
    }
  }();

  for (size_t i = 0; i < lSize; ++i) { _begin[i] = lBestSorted[i]; }

  return _begin + lBestEven;
}


ErrorCode Wald07::runImpl(State &_state) {
  _state.bvh.reserve(_state.mesh.faces.size() * 2 - 1); // Assuming perfect binary tree

  auto lAABBs = boundingVolumesFromMesh(_state.mesh);
  build(begin(lAABBs), end(lAABBs), _state.bvh);

  return ErrorCode::OK;
}
