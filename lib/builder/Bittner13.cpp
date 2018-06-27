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
#include "misc/OMPReductions.hpp"
#include "CUDAHeap.hpp"
#include <algorithm>
#include <chrono>
#include <fmt/format.h>
#include <random>
#include <thread>

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;
using namespace BVHTest::misc;


// Quality of life defines
#define SUM_OF(x) vSumAndMin[x].sum
#define MIN_OF(x) vSumAndMin[x].min

Bittner13::~Bittner13() {}

void Bittner13::fromJSON(const json &_j) {
  OptimizerBase::fromJSON(_j);
  vMaxNumStepps     = _j.value("maxNumStepps", vMaxNumStepps);
  vBatchPercent     = _j.value("batchPercent", vBatchPercent);
  vRandom           = _j.value("random", vRandom);
  vSortBatch        = _j.value("sort", vSortBatch);
  vStrictSequential = _j.value("strictSequential", vStrictSequential);

  if (vBatchPercent <= 0.01f) vBatchPercent = 0.01f;
  if (vBatchPercent >= 75.0f) vBatchPercent = 75.0f;
}

json Bittner13::toJSON() const {
  json lJSON                = OptimizerBase::toJSON();
  lJSON["maxNumStepps"]     = vMaxNumStepps;
  lJSON["batchPercent"]     = vBatchPercent;
  lJSON["random"]           = vRandom;
  lJSON["sort"]             = vSortBatch;
  lJSON["strictSequential"] = vStrictSequential;
  return lJSON;
}

struct FindNodeStruct {
  uint32_t node;
  float    cost;

  inline bool operator<(FindNodeStruct const &_b) const noexcept { return cost > _b.cost; }
};

uint32_t Bittner13::findNodeForReinsertion(uint32_t _n, BVH &_bvh) {
  float           lBestCost      = numeric_limits<float>::infinity();
  uint32_t        lBestNodeIndex = 0;
  AABB            lBBox          = _bvh.bbox(_n);
  float           lSArea         = _bvh.surfaceArea(_n);
  uint32_t        lSize          = 1;
  FindNodeStruct  lPQ[QUEUE_SIZE];
  FindNodeStruct *lBegin = lPQ;

  lPQ[0] = {_bvh.root(), 0.0f};
  while (lSize > 0) {
    FindNodeStruct lCurr = lPQ[0];
    CUDA_pop_heap(lBegin, lBegin + lSize);
    lSize--;

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    AABB lMerge = lBBox;
    lMerge.mergeWith(_bvh.bbox(lCurr.node));

    float lDirectCost = lMerge.surfaceArea();
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = lCurr.node;
    }

    float lNewInduced = lTotalCost - _bvh.surfaceArea(lCurr.node);
    if ((lNewInduced + lSArea) < lBestCost) {
      if (!_bvh.isLeaf(lCurr.node)) {
        lPQ[lSize + 0] = {_bvh.left(lCurr.node), lNewInduced};
        lPQ[lSize + 1] = {_bvh.right(lCurr.node), lNewInduced};
        CUDA_push_heap(lBegin, lBegin + lSize + 1);
        CUDA_push_heap(lBegin, lBegin + lSize + 2);
        lSize += 2;
      }
    }
  }

  return lBestNodeIndex;
}

Bittner13::RM_RES Bittner13::removeNode(uint32_t _node, BVH &_bvh) {
  if (_bvh.isLeaf(_node) || _node == _bvh.root()) { return {false, {0, 0}, {0, 0}}; }

  uint32_t lSibling     = _bvh.sibling(_node);
  uint32_t lLeft        = _bvh.left(_node);
  uint32_t lRight       = _bvh.right(_node);
  uint32_t lParent      = _bvh.parent(_node);
  uint32_t lGrandParent = _bvh.parent(lParent);

  // FREE LIST:    lNode, lParent
  // INSERT LIST:  lLeft, lRight
  // CHANGED LIST: lGrandParent, lSibling

  float lLeftSA  = _bvh.surfaceArea(lLeft);
  float lRightSA = _bvh.surfaceArea(lRight);


  if (lParent == _bvh.root()) { return {false, {0, 0}, {0, 0}}; } // Can not remove node with this algorithm

  // Remove nodes
  if (_bvh.isLeftChild(lParent)) {
    _bvh.left(lGrandParent) = lSibling;
    _bvh.isLeft(lSibling)   = TRUE;
    _bvh.parent(lSibling)   = lGrandParent;
  } else {
    _bvh.right(lGrandParent) = lSibling;
    _bvh.isLeft(lSibling)    = FALSE;
    _bvh.parent(lSibling)    = lGrandParent;
  }

  // update Bounding Boxes
  fixTree(lGrandParent, _bvh);

  if (lLeftSA > lRightSA) {
    return {true, {lLeft, lRight}, {_node, lParent}};
  } else {
    return {true, {lRight, lLeft}, {_node, lParent}};
  }
}


void Bittner13::reinsert(uint32_t _node, uint32_t _unused, BVH &_bvh) {
  uint32_t lBest = findNodeForReinsertion(_node, _bvh);
  uint32_t lRoot = _bvh.parent(lBest);

  if (lBest == _bvh.root()) {
    // Adjust root if needed
    _bvh.setNewRoot(_unused);
    lRoot = _unused;
  } else {
    // Insert the unused node
    if (_bvh.isLeftChild(lBest)) {
      _bvh.left(lRoot)     = _unused;
      _bvh.isLeft(_unused) = TRUE;
    } else {
      _bvh.right(lRoot)    = _unused;
      _bvh.isLeft(_unused) = FALSE;
    }
  }

  // Insert the other nodes
  _bvh.parent(_unused) = lRoot;
  _bvh.left(_unused)   = lBest;
  _bvh.right(_unused)  = _node;

  _bvh.parent(lBest) = _unused;
  _bvh.isLeft(lBest) = TRUE;
  _bvh.parent(_node) = _unused;
  _bvh.isLeft(_node) = FALSE;

  fixTree(_unused, _bvh);
}

void Bittner13::fixTree(uint32_t _node, BVH &_bvh) {
  uint32_t lNode = _node;

  uint32_t lLastIndex        = _bvh.left(lNode);
  bool     lLastWasLeft      = true;
  uint32_t lCurrSiblingIndex = 0;

  AABB  lBBox = _bvh.bbox(lLastIndex);
  float lSum  = SUM_OF(lLastIndex);
  float lMin  = MIN_OF(lLastIndex);
  float lNum  = _bvh.numChildren(lLastIndex);

  float lSArea;


  while (true) {
    lCurrSiblingIndex = lLastWasLeft ? _bvh.right(lNode) : _bvh.left(lNode);

    lBBox.mergeWith(_bvh.bbox(lCurrSiblingIndex));
    lSArea                  = lBBox.surfaceArea();
    _bvh.bbox(lNode)        = lBBox;
    _bvh.surfaceArea(lNode) = lSArea;

    lSum                    = lSum + SUM_OF(lCurrSiblingIndex) + lSArea * getCostInner();
    lMin                    = min(lMin, MIN_OF(lCurrSiblingIndex));
    lNum                    = lNum + _bvh.numChildren(lCurrSiblingIndex) + 2;
    vSumAndMin[lNode]       = {lSum, lMin};
    _bvh.numChildren(lNode) = lNum;

    if (lNode == _bvh.root()) { return; } // We processed the root ==> everything is done

    lLastWasLeft = _bvh.isLeftChild(lNode);
    lLastIndex   = lNode;
    lNode        = _bvh.parent(lNode);
  }
}

void Bittner13::initSumAndMin(BVH &_bvh) {
  vSumAndMin.resize(_bvh.size(), {numeric_limits<float>::infinity(), numeric_limits<float>::infinity()});
  if (_bvh.empty()) { return; }

  __uint128_t lBitStack = 0;
  uint32_t    lNode     = _bvh.root();
  uint32_t    lRoot     = lNode;

  uint32_t lLeft  = 0;
  uint32_t lRight = 0;
  float    lSArea = 0.0f;

  while (true) {
    while (!_bvh.isLeft(lNode)) {
      lBitStack <<= 1;
      lBitStack |= 1;
      lNode = _bvh.left(lNode);
    }

    lSArea = _bvh.surfaceArea(lNode);

    // Leaf
    vSumAndMin[lNode] = {lSArea * (float)getCostTri(), lSArea};

    // Backtrack if left and right children are processed
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0 && lNode == lRoot) { return; } // We are done
      lNode             = _bvh.parent(lNode);
      lLeft             = _bvh.left(lNode);
      lRight            = _bvh.right(lNode);
      lSArea            = _bvh.surfaceArea(lNode);
      vSumAndMin[lNode] = {SUM_OF(lLeft) + SUM_OF(lRight) + lSArea * (float)getCostInner(),
                           min(MIN_OF(lLeft), MIN_OF(lRight))};
      lBitStack >>= 1;
    }

    lNode = _bvh.right(_bvh.parent(lNode));
    lBitStack ^= 1;
  }
}

ErrorCode Bittner13::runMetric(State &_state) {
  typedef tuple<uint32_t, float> TUP;
  vector<TUP>                    lTodoList;
  BVH &                          lBVH = _state.bvh;
  lTodoList.resize(lBVH.size());

  uint32_t lNumNodes  = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(lBVH.size()));
  auto     lComp      = [](TUP const &_l, TUP const &_r) -> bool { return get<1>(_l) > get<1>(_r); };
  uint32_t lNumStepps = vMaxNumStepps;

  if (vStrictSequential) { lNumStepps *= lNumNodes; }

#if ENABLE_PROGRESS_BAR
  uint32_t lCounter = 0;
  uint32_t lSkipp   = 0;
  if (lNumStepps > 100) { lSkipp = lNumStepps / 100; }
#endif

  for (uint32_t i = 0; i < lNumStepps; ++i) {
#if ENABLE_PROGRESS_BAR
    if (lSkipp == 0) {
      progress(fmt::format("METRIC; Stepp {:<12}; SAH: {:<6.6}", i, lBVH.calcSAH()), i, lNumStepps - 1);
    } else {
      if ((lCounter++) > lSkipp) {
        lCounter = 0;
        progress(fmt::format("METRIC; Stepp {:<12}; SAH: {:<6.6}", i, lBVH.calcSAH()), i, lNumStepps - 1);
      }
    }
#endif

    OMP_fi lWorstNode = {0.0f, 0};

    // Select nodes to reinsert
#pragma omp parallel for reduction(maxValF : lWorstNode)
    for (uint32_t j = 0; j < lBVH.size(); ++j) {
      float lSA = lBVH.surfaceArea(j);

      bool lIsRoot       = j == lBVH.root();
      bool lParentIsRoot = lBVH.parent(j) == lBVH.root();
      bool lCanRemove    = !lIsRoot && !lParentIsRoot && !lBVH.isLeaf(j);

      float lCost  = lCanRemove ? ((lSA * lSA * lSA * (float)lBVH.numChildren(j)) / (SUM_OF(j) * MIN_OF(j))) : 0.0f;
      lTodoList[j] = {j, lCost};

      if (lCost > lWorstNode.val) {
        lWorstNode.val = lCost;
        lWorstNode.ind = j;
      }
    }


    if (!vStrictSequential) {
      nth_element(begin(lTodoList), begin(lTodoList) + lNumNodes, end(lTodoList), lComp);
      if (vSortBatch) { sort(begin(lTodoList), begin(lTodoList) + lNumNodes, lComp); }

      // Reinsert nodes
      for (uint32_t j = 0; j < lNumNodes; ++j) {
        auto [lNodeIndex, _]            = lTodoList[j];
        auto [lRes, lTInsert, lTUnused] = removeNode(lNodeIndex, lBVH);
        auto [l1stIndex, l2ndIndex]     = lTInsert;
        auto [lU1, lU2]                 = lTUnused;

        if (!lRes) { continue; }

        reinsert(l1stIndex, lU1, lBVH);
        reinsert(l2ndIndex, lU2, lBVH);
      }
    } else {
      auto [lRes, lTInsert, lTUnused] = removeNode(lWorstNode.ind, lBVH);
      auto [l1stIndex, l2ndIndex]     = lTInsert;
      auto [lU1, lU2]                 = lTUnused;

      if (!lRes) { return ErrorCode::BVH_ERROR; }

      reinsert(l1stIndex, lU1, lBVH);
      reinsert(l2ndIndex, lU2, lBVH);
    }
  }

  return ErrorCode::OK;
}

ErrorCode Bittner13::runRandom(State &_state) {
  vector<uint32_t> lTodoList;
  lTodoList.resize(_state.bvh.size());

  // Init List
#pragma omp parallel for
  for (uint32_t i = 0; i < lTodoList.size(); ++i) { lTodoList[i] = i; }

  uint32_t      lNumNodes  = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  uint32_t      lNumStepps = vMaxNumStepps;
  random_device lRD;
  mt19937       lPRNG(lRD());
  uniform_int_distribution<uint32_t> lDis(0, lTodoList.size() - 1);

  if (vStrictSequential) { lNumStepps *= lNumNodes; }

#if ENABLE_PROGRESS_BAR
  uint32_t lCounter = 0;
  uint32_t lSkipp   = 0;
  if (lNumStepps > 100) { lSkipp = lNumStepps / 100; }
#endif

  for (uint32_t i = 0; i < lNumStepps; ++i) {
#if ENABLE_PROGRESS_BAR
    if (lSkipp == 0) {
      progress(fmt::format("RAND; Stepp {:<12}; SAH: {:<6.6}", i, _state.bvh.calcSAH()), i, lNumStepps - 1);
    } else {
      if ((lCounter++) > lSkipp) {
        lCounter = 0;
        progress(fmt::format("RAND; Stepp {:<12}; SAH: {:<6.6}", i, _state.bvh.calcSAH()), i, lNumStepps - 1);
      }
    }
#endif

    // Select nodes to reinsert
    if (!vStrictSequential) {
      shuffle(begin(lTodoList), end(lTodoList), lPRNG);

      // Reinsert nodes
      for (uint32_t j = 0; j < lNumNodes; ++j) {
        auto [lRes, lTInsert, lTUnused] = removeNode(lTodoList[j], _state.bvh);
        auto [l1stIndex, l2ndIndex]     = lTInsert;
        auto [lU1, lU2]                 = lTUnused;

        if (!lRes) { continue; }

        reinsert(l1stIndex, lU1, _state.bvh);
        reinsert(l2ndIndex, lU2, _state.bvh);
      }
    } else {
      auto [lRes, lTInsert, lTUnused] = removeNode(lDis(lPRNG), _state.bvh);
      auto [l1stIndex, l2ndIndex]     = lTInsert;
      auto [lU1, lU2]                 = lTUnused;

      if (!lRes) { continue; }

      reinsert(l1stIndex, lU1, _state.bvh);
      reinsert(l2ndIndex, lU2, _state.bvh);
    }
  }

  return ErrorCode::OK;
}



ErrorCode Bittner13::runImpl(State &_state) {
  initSumAndMin(_state.bvh);

  ErrorCode lRet;
  if (vRandom) {
    lRet = runRandom(_state);
  } else {
    lRet = runMetric(_state);
  }

  _state.bvh.fixLevels();
  vector<SumMin>().swap(vSumAndMin); // Clear memory of vSumAndMin

  return lRet;
}
