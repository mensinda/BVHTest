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

#include "Bittner13Par.hpp"
#include "misc/OMPReductions.hpp"
#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <queue>
#include <random>
#include <thread>

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;
using namespace BVHTest::misc;


// Quality of life defines
#define SUM_OF(x) get<0>(_sumMin[x])
#define MIN_OF(x) get<1>(_sumMin[x])
#define ATO_OF(x) get<2>(_sumMin[x])

#define NODE _bvh[lNode]
#define PARENT _bvh[NODE.parent]
#define LEFT _bvh[NODE.left]
#define RIGHT _bvh[NODE.right]

Bittner13Par::~Bittner13Par() {}

void Bittner13Par::fromJSON(const json &_j) {
  OptimizerBase::fromJSON(_j);
  vMaxNumStepps = _j.value("maxNumStepps", vMaxNumStepps);
  vBatchPercent = _j.value("batchPercent", vBatchPercent);
  vRandom       = _j.value("random", vRandom);
  vSortBatch    = _j.value("sort", vSortBatch);

  if (vBatchPercent <= 0.01f) vBatchPercent = 0.01f;
  if (vBatchPercent >= 75.0f) vBatchPercent = 75.0f;
}

json Bittner13Par::toJSON() const {
  json lJSON            = OptimizerBase::toJSON();
  lJSON["maxNumStepps"] = vMaxNumStepps;
  lJSON["batchPercent"] = vBatchPercent;
  lJSON["random"]       = vRandom;
  lJSON["sort"]         = vSortBatch;
  return lJSON;
}


uint32_t Bittner13Par::findNodeForReinsertion(uint32_t _n, BVHPatchBittner &_bvh) {
  typedef tuple<uint32_t, float> T1;

  float          lBestCost      = numeric_limits<float>::infinity();
  uint32_t       lBestNodeIndex = 0;
  BVHNode const &lNode          = _bvh[_n];
  float          lSArea         = lNode.surfaceArea;
  auto           lComp          = [](T1 const &_l, T1 const &_r) -> bool { return get<1>(_l) > get<1>(_r); };
  priority_queue<T1, vector<T1>, decltype(lComp)> lPQ(lComp);

  lPQ.push({_bvh.root(), 0.0f});
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

Bittner13Par::RM_RES Bittner13Par::removeNode(uint32_t _node, BVHPatchBittner &_bvh, SumMin *_sumMin) {
  if (_bvh[_node].isLeaf() || _node == _bvh.root()) { return {false, {0, 0}, {0, 0}}; }

  BVHNode &lNode             = _bvh[_node];
  uint32_t lSiblingIndex     = _bvh.sibling(lNode);
  BVHNode &lSibling          = _bvh[lSiblingIndex];
  uint32_t lParentIndex      = lNode.parent;
  BVHNode &lParent           = _bvh[lParentIndex];
  uint32_t lGrandParentIndex = lParent.parent;
  BVHNode &lGrandParent      = _bvh[lGrandParentIndex];

  BVHNode &lLeft  = _bvh[lNode.left];
  BVHNode &lRight = _bvh[lNode.right];

  // FREE LIST:   lNode, lParent
  // INSERT LIST: lLeft, lRight

  float lLeftSA  = lLeft.surfaceArea;
  float lRightSA = lRight.surfaceArea;


  if (lParentIndex == _bvh.root()) { return {false, {0, 0}, {0, 0}}; } // Can not remove node with this algorithm

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
  //   fixTree(lGrandParentIndex, _bvh, _sumMin);

  if (lLeftSA > lRightSA) {
    return {true, {lNode.left, lNode.right}, {_node, lParentIndex}};
  } else {
    return {true, {lNode.right, lNode.left}, {_node, lParentIndex}};
  }
}


void Bittner13Par::reinsert(uint32_t _node, uint32_t _unused, BVHPatchBittner &_bvh, SumMin *_sumMin) {
  uint32_t lBestIndex = findNodeForReinsertion(_node, _bvh);

  BVHNode &lNode      = _bvh[_node];
  BVHNode &lBest      = _bvh[lBestIndex];
  BVHNode &lUnused    = _bvh[_unused];
  uint32_t lRootIndex = lBest.parent;
  BVHNode &lRoot      = _bvh[lRootIndex];

  if (lBestIndex == _bvh.root()) {
    // Adjust root if needed
    _bvh.setNewRoot(_unused);
    lRootIndex = _unused;
  } else {
    // Insert the unused node
    if (lBest.isLeftChild()) {
      lRoot.left     = _unused;
      lUnused.isLeft = TRUE;
    } else {
      lRoot.right    = _unused;
      lUnused.isLeft = FALSE;
    }
  }

  // Insert the other nodes
  lUnused.parent = lRootIndex;
  lUnused.left   = lBestIndex;
  lUnused.right  = _node;

  lBest.parent = _unused;
  lBest.isLeft = TRUE;
  lNode.parent = _unused;
  lNode.isLeft = FALSE;
}

void Bittner13Par::fixTree(uint32_t _node, BVH &_bvh, SumMin *_sumMin) {
  uint32_t lNode = _node;
  while (true) {
    AABB lNewAABB = LEFT.bbox;
    lNewAABB.mergeWith(RIGHT.bbox);
    float lSArea     = lNewAABB.surfaceArea();
    NODE.bbox        = lNewAABB;
    NODE.surfaceArea = lSArea;
    NODE.numChildren = LEFT.numChildren + RIGHT.numChildren + 2;
    SUM_OF(lNode)    = SUM_OF(NODE.left) + SUM_OF(NODE.right) + (lSArea * getCostInner());
    MIN_OF(lNode)    = min(MIN_OF(NODE.left), MIN_OF(NODE.right));
    ATO_OF(lNode)    = 0;

    if (lNode == _bvh.root()) { return; } // We processed the root ==> everything is done

    lNode = NODE.parent;
  }
}

void Bittner13Par::initSumAndMin(BVH &_bvh, SumMin *_sumMin) {
  if (_bvh.empty()) { return; }

  __uint128_t lBitStack = 0;
  uint32_t    lNode     = _bvh.root();

  while (true) {
    while (!NODE.isLeaf()) {
      lBitStack <<= 1;
      lBitStack |= 1;
      lNode = NODE.left;
    }

    // Leaf
    SUM_OF(lNode) = NODE.surfaceArea * getCostTri();
    MIN_OF(lNode) = NODE.surfaceArea;
    ATO_OF(lNode) = 0;

    // Backtrack if left and right children are processed
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0 && lNode == 0) { return; } // We are done
      lNode         = NODE.parent;
      SUM_OF(lNode) = SUM_OF(NODE.left) + SUM_OF(NODE.right) + (NODE.surfaceArea * getCostInner());
      MIN_OF(lNode) = min(MIN_OF(NODE.left), MIN_OF(NODE.right));
      ATO_OF(lNode) = 0;
      lBitStack >>= 1;
    }

    lNode = PARENT.right;
    lBitStack ^= 1;
  }
}

ErrorCode Bittner13Par::runMetric(State &_state, SumMin *_sumMin) {
  typedef tuple<uint32_t, float> TUP;

  uint32_t lNumNodes = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  auto     lComp     = [](TUP const &_l, TUP const &_r) -> bool { return get<1>(_l) > get<1>(_r); };

  BVHPatchBittner lTemp(_state.bvh);


  vector<TUP>             lTodoList;
  vector<BVHPatchBittner> lPatches;
  lTodoList.resize(_state.bvh.size());
  lPatches.resize(lNumNodes, lTemp);

  for (uint32_t i = 0; i < vMaxNumStepps; ++i) {
#if ENABLE_PROGRESS_BAR
    progress(fmt::format("METRIC; Stepp {:<12}; SAH: {:<6.6}", i, _state.bvh.calcSAH()), i, vMaxNumStepps - 1);
#endif

    // Select nodes to reinsert
#pragma omp parallel for
    for (uint32_t j = 0; j < _state.bvh.size(); ++j) {
      BVHNode const &lNode = _state.bvh[j];
      float          lSA   = lNode.surfaceArea;

      bool lIsRoot       = j == _state.bvh.root();
      bool lParentIsRoot = lNode.parent == _state.bvh.root();
      bool lCanRemove    = !lIsRoot && !lParentIsRoot && !lNode.isLeaf();

      float lCost  = lCanRemove ? ((lSA * lSA * lSA * (float)lNode.numChildren) / (SUM_OF(j) * MIN_OF(j))) : 0.0f;
      lTodoList[j] = {j, lCost};
    }


    nth_element(begin(lTodoList), begin(lTodoList) + lNumNodes, end(lTodoList), lComp);
    if (vSortBatch) { sort(begin(lTodoList), begin(lTodoList) + lNumNodes, lComp); }

    // Reinsert nodes
    for (uint32_t j = 0; j < lNumNodes; ++j) {
      auto [lNodeIndex, _]            = lTodoList[j];
      auto [lRes, lTInsert, lTUnused] = removeNode(lNodeIndex, lPatches[j], _sumMin);
      auto [l1stIndex, l2ndIndex]     = lTInsert;
      auto [lU1, lU2]                 = lTUnused;

      if (!lRes) { continue; }

      reinsert(l1stIndex, lU1, lPatches[j], _sumMin);
      reinsert(l2ndIndex, lU2, lPatches[j], _sumMin);
    }
  }

  return ErrorCode::OK;
}

ErrorCode Bittner13Par::runRandom(State &_state, SumMin *_sumMin) {
  uint32_t lNumNodes = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));

  BVHPatchBittner lTemp(_state.bvh);

  vector<uint32_t>        lTodoList;
  vector<BVHPatchBittner> lPatches;
  lTodoList.resize(_state.bvh.size());
  lPatches.resize(lNumNodes, lTemp);

  // Init List
#pragma omp parallel for
  for (uint32_t i = 0; i < lTodoList.size(); ++i) {
    lTodoList[i] = i;
  }

  random_device                      lRD;
  mt19937                            lPRNG(lRD());
  uniform_int_distribution<uint32_t> lDis(0, lTodoList.size() - 1);

  for (uint32_t i = 0; i < vMaxNumStepps; ++i) {
#if ENABLE_PROGRESS_BAR
    progress(fmt::format("RAND; Stepp {:<12}; SAH: {:<6.6}", i, _state.bvh.calcSAH()), i, vMaxNumStepps - 1);
#endif

    // Select nodes to reinsert
    shuffle(begin(lTodoList), end(lTodoList), lPRNG);

    // Reinsert nodes
    for (uint32_t j = 0; j < lNumNodes; ++j) {
      auto [lRes, lTInsert, lTUnused] = removeNode(lTodoList[j], lPatches[j], _sumMin);
      auto [l1stIndex, l2ndIndex]     = lTInsert;
      auto [lU1, lU2]                 = lTUnused;

      if (!lRes) { continue; }

      reinsert(l1stIndex, lU1, lPatches[j], _sumMin);
      reinsert(l2ndIndex, lU2, lPatches[j], _sumMin);
    }
  }

  return ErrorCode::OK;
}



ErrorCode Bittner13Par::runImpl(State &_state) {
  unique_ptr<SumMin[]> lSumMinArr(new SumMin[_state.bvh.size()]);
  initSumAndMin(_state.bvh, lSumMinArr.get());

  ErrorCode lRet;
  if (vRandom) {
    lRet = runRandom(_state, lSumMinArr.get());
  } else {
    lRet = runMetric(_state, lSumMinArr.get());
  }

  _state.bvh.fixLevels();

  return lRet;
}
