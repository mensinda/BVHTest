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
#define NODE _bvh[lNode]
#define PARENT _bvh[NODE->parent]
#define LEFT _bvh[NODE->left]
#define RIGHT _bvh[NODE->right]

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
  BVHNode const * lNode          = _bvh[_n];
  float           lSArea         = lNode->surfaceArea;
  uint32_t        lSize          = 1;
  FindNodeStruct  lPQ[QUEUE_SIZE];
  FindNodeStruct *lBegin = lPQ;

  lPQ[0] = {_bvh.root(), 0.0f};
  while (lSize > 0) {
    FindNodeStruct lCurr     = lPQ[0];
    BVHNode const *lCurrNode = _bvh[lCurr.node];
    CUDA_pop_heap(lBegin, lBegin + lSize);
    lSize--;

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    float lDirectCost = directCost(lNode, lCurrNode);
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = lCurr.node;
    }

    float lNewInduced = lTotalCost - lCurrNode->surfaceArea;
    if ((lNewInduced + lSArea) < lBestCost) {
      if (!lCurrNode->isLeaf()) {
        lPQ[lSize + 0] = {lCurrNode->left, lNewInduced};
        lPQ[lSize + 1] = {lCurrNode->right, lNewInduced};
        CUDA_push_heap(lBegin, lBegin + lSize + 1);
        CUDA_push_heap(lBegin, lBegin + lSize + 2);
        lSize += 2;
      }
    }
  }

  return lBestNodeIndex;
}

Bittner13::RM_RES Bittner13::removeNode(uint32_t _node, BVH &_bvh) {
  if (_bvh[_node]->isLeaf() || _node == _bvh.root()) { return {false, {0, 0}, {0, 0}}; }

  BVHNode *lNode             = _bvh[_node];
  uint32_t lSiblingIndex     = _bvh.sibling(lNode);
  BVHNode *lSibling          = _bvh[lSiblingIndex];
  uint32_t lParentIndex      = lNode->parent;
  BVHNode *lParent           = _bvh[lParentIndex];
  uint32_t lGrandParentIndex = lParent->parent;
  BVHNode *lGrandParent      = _bvh[lGrandParentIndex];

  BVHNode *lLeft  = _bvh[lNode->left];
  BVHNode *lRight = _bvh[lNode->right];

  // FREE LIST:    lNode, lParent
  // INSERT LIST:  lLeft, lRight
  // CHANGED LIST: lGrandParent, lSibling

  float lLeftSA  = lLeft->surfaceArea;
  float lRightSA = lRight->surfaceArea;


  if (lParentIndex == _bvh.root()) { return {false, {0, 0}, {0, 0}}; } // Can not remove node with this algorithm

  // Remove nodes
  if (lParent->isLeftChild()) {
    lGrandParent->left = lSiblingIndex;
    lSibling->isLeft   = TRUE;
    lSibling->parent   = lGrandParentIndex;
  } else {
    lGrandParent->right = lSiblingIndex;
    lSibling->isLeft    = FALSE;
    lSibling->parent    = lGrandParentIndex;
  }

  // update Bounding Boxes
  fixTree(lGrandParentIndex, _bvh);

  if (lLeftSA > lRightSA) {
    return {true, {lNode->left, lNode->right}, {_node, lParentIndex}};
  } else {
    return {true, {lNode->right, lNode->left}, {_node, lParentIndex}};
  }
}


void Bittner13::reinsert(uint32_t _node, uint32_t _unused, BVH &_bvh) {
  uint32_t lBestIndex = findNodeForReinsertion(_node, _bvh);

  BVHNode *lNode      = _bvh[_node];
  BVHNode *lBest      = _bvh[lBestIndex];
  BVHNode *lUnused    = _bvh[_unused];
  uint32_t lRootIndex = lBest->parent;
  BVHNode *lRoot      = _bvh[lRootIndex];

  if (lBestIndex == _bvh.root()) {
    // Adjust root if needed
    _bvh.setNewRoot(_unused);
    lRootIndex = _unused;
  } else {
    // Insert the unused node
    if (lBest->isLeftChild()) {
      lRoot->left     = _unused;
      lUnused->isLeft = TRUE;
    } else {
      lRoot->right    = _unused;
      lUnused->isLeft = FALSE;
    }
  }

  // Insert the other nodes
  lUnused->parent = lRootIndex;
  lUnused->left   = lBestIndex;
  lUnused->right  = _node;

  lBest->parent = _unused;
  lBest->isLeft = TRUE;
  lNode->parent = _unused;
  lNode->isLeft = FALSE;

  fixTree(_unused, _bvh);
}

void Bittner13::fixTree(uint32_t _node, BVH &_bvh) {
  uint32_t lNode = _node;

  bool     lLastWasLeft      = true;
  uint32_t lCurrSiblingIndex = 0;
  BVHNode *lCurrSibling      = nullptr;

  AABB lBBox = LEFT->bbox;

  while (true) {
    lCurrSiblingIndex = lLastWasLeft ? NODE->right : NODE->left;
    lCurrSibling      = _bvh[lCurrSiblingIndex];

    lBBox.mergeWith(lCurrSibling->bbox);
    NODE->bbox        = lBBox;
    NODE->surfaceArea = lBBox.surfaceArea();

    if (lNode == _bvh.root()) { return; } // We processed the root ==> everything is done

    lLastWasLeft = NODE->isLeftChild();
    lNode        = NODE->parent;
  }
}

ErrorCode Bittner13::runMetric(State &_state) {
  typedef tuple<uint32_t, float> TUP;
  vector<TUP>                    lTodoList;
  lTodoList.resize(_state.bvh.size());

  uint32_t lNumNodes  = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  auto     lComp      = [](TUP const &_l, TUP const &_r) -> bool { return get<1>(_l) > get<1>(_r); };
  uint32_t lNumStepps = vMaxNumStepps;

  if (vStrictSequential) { lNumStepps *= lNumNodes; }

#if ENABLE_PROGRESS_BAR
  uint32_t lCounter = 0;
  uint32_t lSkipp   = 0;
  if (lNumStepps > 100) { lSkipp = lNumStepps / 100; }
#endif

  benchmarkInitData(_state, [&]() -> float { return _state.bvh.calcSAH(); });

  for (uint32_t i = 0; i < lNumStepps; ++i) {
#if ENABLE_PROGRESS_BAR
    if (lSkipp == 0) {
      progress(fmt::format("METRIC; Stepp {:<12}; SAH: {:<6.6}", i, _state.bvh.calcSAH()), i, lNumStepps - 1);
    } else {
      if ((lCounter++) > lSkipp) {
        lCounter = 0;
        progress(fmt::format("METRIC; Stepp {:<12}; SAH: {:<6.6}", i, _state.bvh.calcSAH()), i, lNumStepps - 1);
      }
    }
#endif

    benchmarkStartTimer(_state);

    OMP_fi lWorstNode = {0.0f, 0};

    // Select nodes to reinsert
#pragma omp parallel for reduction(maxValF : lWorstNode)
    for (uint32_t j = 0; j < _state.bvh.size(); ++j) {
      BVHNode const *lNode = _state.bvh[j];
      float          lSA   = lNode->surfaceArea;

      if (lNode->isLeaf()) {
        lTodoList[j] = {j, 0};
        continue;
      }

      float lLeftSA  = _state.bvh[lNode->left]->surfaceArea;
      float lRightSA = _state.bvh[lNode->right]->surfaceArea;

      float lCost  = (lSA * lSA * lSA * 2.0f) / ((lLeftSA + lRightSA) * min(lLeftSA, lRightSA));
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
        auto [lRes, lTInsert, lTUnused] = removeNode(lNodeIndex, _state.bvh);
        auto [l1stIndex, l2ndIndex]     = lTInsert;
        auto [lU1, lU2]                 = lTUnused;

        if (!lRes) { continue; }

        reinsert(l1stIndex, lU1, _state.bvh);
        reinsert(l2ndIndex, lU2, _state.bvh);
      }
    } else {
      auto [lRes, lTInsert, lTUnused] = removeNode(lWorstNode.ind, _state.bvh);
      auto [l1stIndex, l2ndIndex]     = lTInsert;
      auto [lU1, lU2]                 = lTUnused;

      if (!lRes) { return ErrorCode::BVH_ERROR; }

      reinsert(l1stIndex, lU1, _state.bvh);
      reinsert(l2ndIndex, lU2, _state.bvh);
    }

    benchmarkRecordData(_state, [&]() -> float { return _state.bvh.calcSAH(); });
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

  benchmarkInitData(_state, [&]() -> float { return _state.bvh.calcSAH(); });

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

    benchmarkStartTimer(_state);

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

    benchmarkRecordData(_state, [&]() -> float { return _state.bvh.calcSAH(); });
  }

  return ErrorCode::OK;
}



ErrorCode Bittner13::runImpl(State &_state) {
  if (vRandom) { return runRandom(_state); }

  return runMetric(_state);
}

void Bittner13::teardown(State &_state) { _state.bvh.fixLevels(); }
