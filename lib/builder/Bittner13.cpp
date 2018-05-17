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

uint32_t Bittner13::findNodeForReinsertion(uint32_t _n, vector<BVH> &_bvh) {
  float      lBestCost      = numeric_limits<float>::infinity();
  uint32_t   lBestNodeIndex = 0;
  BVH const &lNode          = _bvh[_n];
  float      lSArea         = lNode.bbox.surfaceArea();
  auto       lComp          = [](T1 const &_l, T1 const &_r) -> bool { return get<1>(_l) > get<1>(_r); };
  priority_queue<T1, vector<T1>, decltype(lComp)> lPQ(lComp);

  lPQ.push({0, 0.0f});
  while (!lPQ.empty()) {
    auto [lCurrNodeIndex, lCurrCost] = lPQ.top();
    BVH const &lCurrNode             = _bvh[lCurrNodeIndex];
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

inline tuple<float, float, uint32_t> travNode(uint32_t _n, vector<BVH> &_bvh) {
  BVH const &lNode   = _bvh[_n];
  float      lNodeSA = lNode.bbox.surfaceArea();

  if (lNode.isLeaf()) { return {lNodeSA, lNodeSA, 1}; }

  auto [lMin, lSum, lNum] = travNode(lNode.left, _bvh);
  auto [rMin, rSum, rNum] = travNode(lNode.right, _bvh);

  return {min(lMin, rMin), lSum + rSum, lNum + rNum};
}

float Bittner13::mComb(uint32_t _n, vector<BVH> &_bvh) {
  float lSum = 1.0f;
  float lMin = 1.0f;

  BVH const &lNode   = _bvh[_n];
  float      lNodeSA = lNode.bbox.surfaceArea();

  if (!lNode.isLeaf()) {
    auto [lSumSA, lMinSA, lNumChilds] = travNode(_n, _bvh);

    lSum = lNodeSA / ((1.0f / static_cast<float>(lNumChilds)) * lSumSA);
    lMin = lMinSA;
  }

  return lSum * lMin * lNodeSA;
}

#if 0
void printBVH(uint32_t _node, vector<BVH> const &_bvh, string _name) {
  auto       lLogger = getLogger();
  BVH const &lNode   = _bvh[_node];
  lLogger->info("Node: {}", _name);
  lLogger->info("  - Index:   {}", _node);
  lLogger->info("  - Parent:  {}", lNode.parent);
  lLogger->info("  - Sibling: {}", lNode.sibling);
  lLogger->info("  - Left:    {}", lNode.left);
  lLogger->info("  - Right:   {}", lNode.right);
  lLogger->info("");
}
#endif

void Bittner13::fixBBOX(uint32_t _node, vector<BVH> &_bvh) {
  uint32_t lCurrNode = _node;
  while (lCurrNode != 0) {
    BVH &lCurr    = _bvh[lCurrNode];
    AABB lNewAABB = _bvh[lCurr.left].bbox;
    lNewAABB.mergeWith(_bvh[lCurr.right].bbox);
    lCurr.bbox = lNewAABB;
    lCurrNode  = lCurr.parent;
  }
}


void Bittner13::reinsert(uint32_t _node, uint32_t _unused, vector<BVH> &_bvh) {
  uint32_t lBestIndex = findNodeForReinsertion(_node, _bvh);
  if (lBestIndex == 0) {
    getLogger()->error("Bittner13: Can not reinsert at the root!");
    return;
  }

  BVH &    lNode      = _bvh[_node];
  BVH &    lBest      = _bvh[lBestIndex];
  BVH &    lUnused    = _bvh[_unused];
  uint32_t lRootIndex = lBest.parent;
  BVH &    lRoot      = _bvh[lRootIndex];

  // Insert the unused node
  if (lRoot.left == lBestIndex) {
    lRoot.left      = _unused;
    lUnused.sibling = lRoot.right;
  } else if (lRoot.right == lBestIndex) {
    lRoot.right     = _unused;
    lUnused.sibling = lRoot.left;
  } else {
    getLogger()->error("Bittner13: Invalid BVH tree! Can not reinsert!");
    return;
  }

  _bvh[lRoot.left].sibling  = lRoot.right;
  _bvh[lRoot.right].sibling = lRoot.left;
  _bvh[lRoot.left].parent   = lRootIndex;
  _bvh[lRoot.right].parent  = lRootIndex;

  // Insert the other nodes
  lUnused.left  = lBestIndex;
  lUnused.right = _node;

  lBest.sibling = _node;
  lNode.sibling = lBestIndex;

  lBest.parent = _unused;
  lNode.parent = _unused;

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

      BVH &    lNode             = _state.bvh[lNodeIndex];
      uint32_t lParentIndex      = lNode.parent;
      BVH &    lParent           = _state.bvh[lParentIndex];
      uint32_t lGrandParentIndex = lParent.parent;
      BVH &    lGrandParent      = _state.bvh[lGrandParentIndex];

      BVH &lLeft  = _state.bvh[lNode.left];
      BVH &lRight = _state.bvh[lNode.right];

      float lLeftSA  = lLeft.bbox.surfaceArea();
      float lRightSA = lRight.bbox.surfaceArea();


      if (lParentIndex == 0) { continue; } // Can not remove node with this algorithm

      // Remove nodes
      if (lGrandParent.left == lParentIndex) {
        lGrandParent.left = lNode.sibling;
      } else if (lGrandParent.right == lParentIndex) {
        lGrandParent.right = lNode.sibling;
      } else {
        lLogger->error("Bittner13: Invalid BVH tree");
        return ErrorCode::BVH_ERROR;
      }

      _state.bvh[lGrandParent.left].sibling  = lGrandParent.right;
      _state.bvh[lGrandParent.right].sibling = lGrandParent.left;
      _state.bvh[lGrandParent.left].parent   = lGrandParentIndex;
      _state.bvh[lGrandParent.right].parent  = lGrandParentIndex;


      // update Bounding Boxes
      fixBBOX(lGrandParentIndex, _state.bvh);

      // Insert nodes
      uint32_t lFirstIndex  = lLeftSA > lRightSA ? lNode.left : lNode.right;
      uint32_t lSecondIndex = lLeftSA <= lRightSA ? lNode.left : lNode.right;

#if 0
      lLeft.parent   = UINT32_MAX;
      lLeft.sibling  = UINT32_MAX;
      lRight.parent  = UINT32_MAX;
      lRight.sibling = UINT32_MAX;

      lNode.parent  = UINT32_MAX;
      lNode.sibling = UINT32_MAX;
      lNode.left    = UINT32_MAX;
      lNode.right   = UINT32_MAX;

      lParent.parent  = UINT32_MAX;
      lParent.sibling = UINT32_MAX;
      lParent.left    = UINT32_MAX;
      lParent.right   = UINT32_MAX;
#endif

      reinsert(lFirstIndex, lNodeIndex, _state.bvh);
      reinsert(lSecondIndex, lParentIndex, _state.bvh);
    }
  }

  cout << "\x1b[2K\x1b[1G" << flush;

  return ErrorCode::OK;
}
