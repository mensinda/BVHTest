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

#include "Bittner13.hpp"
#include <algorithm>
#include <chrono>
#include <iostream>
#include <queue>
#include <thread>

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;


// Quality of life defines
#define SUM_OF(x) get<0>(vSumAndMin[x])
#define MIN_OF(x) get<1>(vSumAndMin[x])

#define NODE _bvh[lNode]
#define PARENT _bvh[NODE.parent]
#define LEFT _bvh[NODE.left]
#define RIGHT _bvh[NODE.right]

Bittner13::~Bittner13() {}

void Bittner13::fromJSON(const json &_j) {
  OptimizerBase::fromJSON(_j);
  vMaxNumStepps = _j.value("maxNumStepps", vMaxNumStepps);
  vBatchPercent = _j.value("batchPercent", vBatchPercent);

  if (vBatchPercent <= 0.1f) vBatchPercent = 0.1f;
  if (vBatchPercent >= 75.0f) vBatchPercent = 75.0f;
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
  float          lSArea         = lNode.surfaceArea;
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

    float lNewInduced = lTotalCost - lCurrNode.surfaceArea;
    if ((lNewInduced + lSArea) < lBestCost) {
      if (!lCurrNode.isLeaf()) {
        lPQ.push({lCurrNode.left, lNewInduced});
        lPQ.push({lCurrNode.right, lNewInduced});
      }
    }
  }

  return lBestNodeIndex;
}

float Bittner13::mComb(uint32_t _n, BVH &_bvh) {
  BVHNode &lNode   = _bvh[_n];
  float    lNodeSA = lNode.surfaceArea;

  if (lNode.isLeaf()) { return 0.0f; }

#if 0
  float lSum = lNodeSA / (SUM_OF(_n) / static_cast<float>(lNode.numChildren));
  float lMin = lNodeSA / MIN_OF(_n);

  return lSum * lMin * lNodeSA;
#else
  /*
   * $ \frac{lNodeSA}{\frac{1}{lNode.numChildren} * SUM_OF(_n)} * \frac{lNodeSA}{MIN_OF(_n)} * lNodeSA $
   *
   * can be simplified to:
   *
   * $ \frac{ lNodeSA^3 * lNode.numChildren }{ SUM_OF(_n) * MIN_OF(_n) } $
   */
  return (lNodeSA * lNodeSA * lNodeSA * static_cast<float>(lNode.numChildren)) / (SUM_OF(_n) * MIN_OF(_n));
#endif
}


void Bittner13::fixTree(uint32_t _node, BVH &_bvh) {
  uint32_t lNode = _node;
  while (true) {
    AABB lNewAABB = LEFT.bbox;
    lNewAABB.mergeWith(RIGHT.bbox);
    NODE.bbox         = lNewAABB;
    NODE.surfaceArea  = lNewAABB.surfaceArea();
    NODE.numChildren  = LEFT.numChildren + RIGHT.numChildren + 2;
    vSumAndMin[lNode] = {SUM_OF(NODE.left) + SUM_OF(NODE.right), min(MIN_OF(NODE.left), MIN_OF(NODE.right))};

    if (lNode == 0) { return; } // We processed the root ==> everything is done

    lNode = NODE.parent;
  }
}


bool Bittner13::reinsert(uint32_t _node, uint32_t _unused, BVH &_bvh) {
  uint32_t lBestIndex = findNodeForReinsertion(_node, _bvh);
  if (lBestIndex == 0) { return false; }

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

  fixTree(_unused, _bvh);
  return true;
}

void Bittner13::initSumAndMin(BVH &_bvh) {
  vSumAndMin.resize(_bvh.size(), {numeric_limits<float>::infinity(), numeric_limits<float>::infinity()});
  if (_bvh.empty()) { return; }

  __uint128_t lBitStack = 0;
  uint32_t    lNode     = 0;

  while (true) {
    while (!NODE.isLeaf()) {
      lBitStack <<= 1;
      lBitStack |= 1;
      lNode = NODE.left;
    }

    // Leaf
    vSumAndMin[lNode] = {0.0f, NODE.surfaceArea};

    // Backtrack if left and right children are processed
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0 && lNode == 0) { return; } // We are done
      lNode             = NODE.parent;
      vSumAndMin[lNode] = {SUM_OF(NODE.left) + SUM_OF(NODE.right), min(MIN_OF(NODE.left), MIN_OF(NODE.right))};
      lBitStack >>= 1;
    }

    lNode = PARENT.right;
    lBitStack ^= 1;
  }
}


ErrorCode Bittner13::runImpl(State &_state) {
  typedef tuple<uint32_t, float> TUP;
  vector<TUP>                    lTodoList;
  lTodoList.resize(_state.bvh.size());

  initSumAndMin(_state.bvh);

  uint32_t lNumNodes = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  auto     lComp     = [](TUP const &_l, TUP const &_r) -> bool { return get<1>(_l) > get<1>(_r); };

  for (uint32_t i = 0; i < vMaxNumStepps; ++i) {
    cout << "\x1b[2K\x1b[1GProgress: " << (int)(((float)i / vMaxNumStepps) * 100.0f) << "%" << flush;

// Select nodes to reinsert
#pragma omp parallel for
    for (uint32_t j = 0; j < _state.bvh.size(); ++j) {
      float lCost  = mComb(j, _state.bvh);
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

      float lLeftSA  = lLeft.surfaceArea;
      float lRightSA = lRight.surfaceArea;


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
      fixTree(lGrandParentIndex, _state.bvh);

      // Insert nodes
      uint32_t lFirstIndex  = lLeftSA > lRightSA ? lNode.left : lNode.right;
      uint32_t lSecondIndex = lLeftSA <= lRightSA ? lNode.left : lNode.right;

      if (!reinsert(lFirstIndex, lNodeIndex, _state.bvh)) { return ErrorCode::BVH_ERROR; }
      if (!reinsert(lSecondIndex, lParentIndex, _state.bvh)) { return ErrorCode::BVH_ERROR; }
    }
  }

  cout << "\x1b[2K\x1b[1G" << flush;
  _state.bvh.fixLevels();

  vector<SumMin>().swap(vSumAndMin); // Clear memory of vSumAndMin

  return ErrorCode::OK;
}
