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
#include "Bittner13.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <thread>
#include <tuple>

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;

Bittner13::~Bittner13() {}

void Bittner13::fromJSON(const json &_j) {
  OptimizerBase::fromJSON(_j);
  vMaxNumStepps = _j.value("maxNumStepps", vMaxNumStepps);
  vBatchPercent = _j.value("batchPercent", vBatchPercent);

  if (vBatchPercent <= 0.5f || vBatchPercent >= 75.0f) vBatchPercent = 1.0f;
}

json Bittner13::toJSON() const {
  json lJSON            = OptimizerBase::toJSON();
  lJSON["maxNumStepps"] = vMaxNumStepps;
  lJSON["batchPercent"] = vBatchPercent;
  return lJSON;
}


struct NodeAndCost {
  uint32_t node;
  float    cost;
};

typedef tuple<uint32_t, float> T1;

uint32_t Bittner13::findNodeForReinsertion(uint32_t _n, BVH &_bvh) {
  float          lBestCost      = numeric_limits<float>::infinity();
  uint32_t       lBestNodeIndex = 0;
  BVHNode const &lNode          = _bvh[_n];
  float          lSArea         = lNode.bbox.surfaceArea();
  auto           lComp          = [](T1 const &_l, T1 const &_r) -> bool { return get<1>(_l) > get<1>(_r); };
  priority_queue<T1, vector<T1>, decltype(lComp)> lPQ(lComp);

  lPQ.push({0, 0.0f});
  while (!lPQ.empty()) {
    auto [lCurrNodeIndex, lCurrCost] = lPQ.top();
    BVHNode const &lCurrNode         = _bvh[lCurrNodeIndex];
    lPQ.pop();

    if ((lCurrCost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    float lDirectCost = directCost(lNode, lCurrNode);
    float lTotalCost  = lCurrCost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = lCurrNodeIndex;
    }

    float lNewInduced = lTotalCost - lCurrNode.bbox.surfaceArea();
    if (lNewInduced + lSArea < lBestCost) {
      if (!lCurrNode.isLeaf()) {
        lPQ.push({lCurrNode.left, lNewInduced});
        lPQ.push({lCurrNode.right, lNewInduced});
      }
    }
  }

  return lBestNodeIndex;
}

inline tuple<float, float, uint32_t> travNode(uint32_t _n, BVH &_bvh) {
  BVHNode const &lNode   = _bvh[_n];
  float          lNodeSA = lNode.bbox.surfaceArea();

  if (lNode.isLeaf()) { return {lNodeSA, lNodeSA, 1}; }

  auto [lMin, lSum, lNum] = travNode(lNode.left, _bvh);
  auto [rMin, rSum, rNum] = travNode(lNode.right, _bvh);

  return {min(lMin, rMin), lSum + rSum, lNum + rNum};
}

float Bittner13::mComb(uint32_t _n, BVH &_bvh) {
  float lSum = 1.0f;
  float lMin = 1.0f;

  BVHNode const &lNode   = _bvh[_n];
  float          lNodeSA = lNode.bbox.surfaceArea();

  if (!lNode.isLeaf()) {
    auto [lSumSA, lMinSA, lNumChilds] = travNode(_n, _bvh);

    lSum = lNodeSA / ((1.0f / static_cast<float>(lNumChilds)) * lSumSA);
    lMin = lMinSA;
  }

  return lSum * lMin * lNodeSA;
}


void Bittner13::fixBBOX(uint32_t _node, BVH &_bvh) {
  uint32_t lCurrNode = _node;
  while (lCurrNode != 0) {
    BVHNode &lCurr    = _bvh[lCurrNode];
    AABB     lNewAABB = _bvh[lCurr.left].bbox;
    lNewAABB.mergeWith(_bvh[lCurr.right].bbox);
    lCurr.bbox = lNewAABB;
    lCurrNode  = lCurr.parent;
  }
}


void Bittner13::reinsert(uint32_t _node, uint32_t _unused, BVH &_bvh) {
  uint32_t lBestIndex = findNodeForReinsertion(_node, _bvh);
  if (lBestIndex == 0) {
    getLogger()->error("Bittner13: Can not reinsert at the root!");
    return;
  }

  BVHNode &lNode      = _bvh[_node];
  BVHNode &lBest      = _bvh[lBestIndex];
  BVHNode &lUnused    = _bvh[_unused];
  uint32_t lRootIndex = lBest.parent;
  BVHNode &lRoot      = _bvh[lRootIndex];

  // Insert the unused node
  if (lBest.isLeftChild()) {
    lRoot.left     = _unused;
    lUnused.isLeft = TRUE;
  } else {
    lRoot.right    = _unused;
    lUnused.isLeft = FALSE;
  }

  // Insert the other nodes
  lUnused.parent = lRootIndex;
  lUnused.left   = lBestIndex;
  lUnused.right  = _node;

  lBest.parent = _unused;
  lBest.isLeft = TRUE;
  lNode.parent = _unused;
  lNode.isLeft = FALSE;

  fixBBOX(_unused, _bvh);
}


ErrorCode Bittner13::runImpl(State &_state) {
  auto                           lLogger = getLogger();
  typedef tuple<uint32_t, float> TUP;
  vector<TUP>                    lTodoList;
  lTodoList.resize(_state.bvh.size());

  uint32_t lNumNodes = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  auto     lComp     = [](TUP const &_l, TUP const &_r) -> bool { return get<1>(_l) > get<1>(_r); };

  for (uint32_t i = 0; i < vMaxNumStepps; ++i) {
    //     lLogger->info("RUN: {}", i);
    cout << fmt::format("\x1b[2K\x1b[1GProgress: {}%",
                        static_cast<int>((static_cast<float>(i) / vMaxNumStepps) * 100.0f))
         << flush;

// Select nodes to reinsert
#pragma omp parallel for
    for (uint32_t j = 0; j < _state.bvh.size(); ++j) {
      float lCost  = _state.bvh[j].isLeaf() ? 0 : mComb(j, _state.bvh);
      lTodoList[j] = {j, lCost};
    }

    nth_element(begin(lTodoList), begin(lTodoList) + lNumNodes, end(lTodoList), lComp);

    // Reinsert nodes
    for (uint32_t j = 0; j < lNumNodes; ++j) {
      auto [lNodeIndex, _] = lTodoList[j];
      if (_state.bvh[lNodeIndex].isLeaf() || lNodeIndex == 0) continue; // Theoretically should never happen

      BVHNode &lNode             = _state.bvh[lNodeIndex];
      uint32_t lSiblingIndex     = _state.bvh.sibling(lNode);
      BVHNode &lSibling          = _state.bvh[lSiblingIndex];
      uint32_t lParentIndex      = lNode.parent;
      BVHNode &lParent           = _state.bvh[lParentIndex];
      uint32_t lGrandParentIndex = lParent.parent;
      BVHNode &lGrandParent      = _state.bvh[lGrandParentIndex];

      BVHNode &lLeft  = _state.bvh[lNode.left];
      BVHNode &lRight = _state.bvh[lNode.right];

      // FREE LIST:   lNode, lParent
      // INSERT LIST: lLeft, lRight

      float lLeftSA  = lLeft.bbox.surfaceArea();
      float lRightSA = lRight.bbox.surfaceArea();


      if (lParentIndex == 0) { continue; } // Can not remove node with this algorithm

      // Remove nodes
      if (lParent.isLeftChild()) {
        lGrandParent.left = lSiblingIndex;
        lSibling.isLeft   = TRUE;
        lSibling.parent   = lGrandParentIndex;
      } else {
        lGrandParent.right = lSiblingIndex;
        lSibling.isLeft    = FALSE;
        lSibling.parent    = lGrandParentIndex;
      }

      // update Bounding Boxes
      fixBBOX(lGrandParentIndex, _state.bvh);

      // Insert nodes
      uint32_t lFirstIndex  = lLeftSA > lRightSA ? lNode.left : lNode.right;
      uint32_t lSecondIndex = lLeftSA <= lRightSA ? lNode.left : lNode.right;

      reinsert(lFirstIndex, lNodeIndex, _state.bvh);
      reinsert(lSecondIndex, lParentIndex, _state.bvh);
    }
  }

  cout << "\x1b[2K\x1b[1G" << flush;

  return ErrorCode::OK;
}
