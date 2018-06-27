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
#include "Bittner13Par.hpp"
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
#define SUM_OF(x) _sumMin[x].sum
#define MIN_OF(x) _sumMin[x].min

#define NODE _bvh[lNode]
#define PARENT _bvh[NODE->parent]
#define LEFT _bvh[NODE->left]
#define RIGHT _bvh[NODE->right]

Bittner13Par::~Bittner13Par() {}

void Bittner13Par::fromJSON(const json &_j) {
  OptimizerBase::fromJSON(_j);
  vMaxNumStepps = _j.value("maxNumStepps", vMaxNumStepps);
  vNumChunks    = _j.value("numChunks", vNumChunks);
  vAltFNQSize   = _j.value("altFindNodeQueueSize", vAltFNQSize);
  vBatchPercent = _j.value("batchPercent", vBatchPercent);
  vRandom       = _j.value("random", vRandom);
  vSortBatch    = _j.value("sort", vSortBatch);
  vShuffleList  = _j.value("shuffle", vShuffleList);
  vOffsetAccess = _j.value("offsetAccess", vOffsetAccess);
  vAltFindNode  = _j.value("altFindNode", vAltFindNode);

  if (vAltFNQSize <= 4) vAltFNQSize = 4;
  if (vBatchPercent <= 0.01f) vBatchPercent = 0.01f;
  if (vBatchPercent >= 75.0f) vBatchPercent = 75.0f;
}

json Bittner13Par::toJSON() const {
  json lJSON                    = OptimizerBase::toJSON();
  lJSON["maxNumStepps"]         = vMaxNumStepps;
  lJSON["numChunks"]            = vNumChunks;
  lJSON["altFindNodeQueueSize"] = vAltFNQSize;
  lJSON["batchPercent"]         = vBatchPercent;
  lJSON["random"]               = vRandom;
  lJSON["sort"]                 = vSortBatch;
  lJSON["shuffle"]              = vShuffleList;
  lJSON["offsetAccess"]         = vOffsetAccess;
  lJSON["altFindNode"]          = vAltFindNode;
  return lJSON;
}

struct HelperStruct {
  uint32_t node;
  float    cost;
  uint32_t level;

  inline bool operator<(HelperStruct const &_b) const noexcept { return cost > _b.cost; }
};


Bittner13Par::NodeLevel Bittner13Par::findNode1(uint32_t _n, BVHPatch &_bvh) {
  float         lBestCost      = HUGE_VALF;
  NodeLevel     lBestNodeIndex = {0, 0};
  AABB const &  lNodeBBox      = _bvh.orig_bbox(_n);
  float         lSArea         = _bvh.orig_surfaceArea(_n);
  uint32_t      lSize          = 1;
  HelperStruct  lPQ[QUEUE_SIZE];
  HelperStruct *lBegin = lPQ;

  lPQ[0] = {_bvh.root(), 0.0f, 0};
  while (lSize > 0) {
    HelperStruct lCurr     = lPQ[0];
    uint64_t     lCurrNode = _bvh.getSubset(lCurr.node);
    auto         lBBox     = _bvh.getAABB(lCurr.node, lCurr.level);
    CUDA_pop_heap(lBegin, lBegin + lSize);
    lSize--;

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    lBBox.box.mergeWith(lNodeBBox);
    float lDirectCost = lBBox.box.surfaceArea();
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = {lCurr.node, lCurr.level};
    }

    float lNewInduced = lTotalCost - lBBox.sarea;
    if ((lNewInduced + lSArea) < lBestCost) {
      if (!_bvh.isLeft(lCurrNode)) {
        assert(lSize + 2 < QUEUE_SIZE);
        lPQ[lSize + 0] = {_bvh.left(lCurrNode), lNewInduced, lCurr.level + 1};
        lPQ[lSize + 1] = {_bvh.right(lCurrNode), lNewInduced, lCurr.level + 1};
        CUDA_push_heap(lBegin, lBegin + lSize + 1);
        CUDA_push_heap(lBegin, lBegin + lSize + 2);
        lSize += 2;
      }
    }
  }

  return lBestNodeIndex;
}

Bittner13Par::NodeLevel Bittner13Par::findNode2(uint32_t _n, BVHPatch &_bvh) {
  float        lBestCost      = HUGE_VALF;
  NodeLevel    lBestNodeIndex = {0, 0};
  AABB const & lNodeBBox      = _bvh.orig_bbox(_n);
  float        lSArea         = _bvh.orig_surfaceArea(_n);
  float        lMin           = 0.0f;
  float        lMax           = HUGE_VALF;
  uint32_t     lMinIndex      = 0;
  uint32_t     lMaxIndex      = 1;
  HelperStruct lPQ[vAltFNQSize];
  HelperStruct lCurr;

  // Init
  for (uint32_t i = 0; i < vAltFNQSize; ++i) { lPQ[i].cost = HUGE_VALF; }

  lPQ[0] = {_bvh.root(), 0.0f, 0};
  while (lMin < HUGE_VALF) {
    lCurr               = lPQ[lMinIndex];
    lPQ[lMinIndex].cost = HUGE_VALF;
    uint64_t lCurrNode  = _bvh.getSubset(lCurr.node);
    auto     lBBox      = _bvh.getAABB(lCurr.node, lCurr.level);

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    lBBox.box.mergeWith(lNodeBBox);
    float lDirectCost = lBBox.box.surfaceArea();
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = {lCurr.node, lCurr.level};
    }

    float lNewInduced = lTotalCost - lBBox.sarea;
    if ((lNewInduced + lSArea) < lBestCost && !_bvh.isLeft(lCurrNode)) {
      lPQ[lMinIndex] = {_bvh.left(lCurrNode), lNewInduced, lCurr.level + 1};
      lPQ[lMaxIndex] = {_bvh.right(lCurrNode), lNewInduced, lCurr.level + 1};
    }

    lMin = HUGE_VALF;
    lMax = 0.0f;
    for (uint32_t i = 0; i < vAltFNQSize; ++i) {
      if (lPQ[i].cost < lMin) {
        lMin      = lPQ[i].cost;
        lMinIndex = i;
      }
      if (lPQ[i].cost > lMax) {
        lMax      = lPQ[i].cost;
        lMaxIndex = i;
      }
    }
  }

  return lBestNodeIndex;
}

#define SPINN_LOCK(N)                                                                                                  \
  while (_sumMin[N].flag.test_and_set(memory_order_acquire)) { this_thread::yield(); }
#define IF_NOT_LOCK(N) if (_sumMin[N].flag.test_and_set(memory_order_acquire))
#define RELEASE_LOCK(N) _sumMin[N].flag.clear(memory_order_release);

Bittner13Par::RM_RES Bittner13Par::removeNode(uint32_t _node, BVHPatch &_bvh, SumMin *_sumMin) {
  RM_RES lFalse = {false, {0, 0}, {0, 0}, {0, 0}};
  if (_bvh.orig_isLeaf(_node) || _node == _bvh.root()) { return lFalse; }

  IF_NOT_LOCK(_node) { return lFalse; }

  uint16_t lNode         = _bvh.patchNode(_node, PINDEX_NODE);
  uint32_t lParentIndex  = _bvh.patch_parent(lNode);
  uint32_t lLeftIndex    = _bvh.patch_left(lNode);
  uint32_t lRightIndex   = _bvh.patch_right(lNode);
  uint32_t lSiblingIndex = _bvh.isRightChild(lNode) ? _bvh.orig_left(lParentIndex) : _bvh.orig_right(lParentIndex);

  if (lParentIndex == _bvh.root()) {
    RELEASE_LOCK(_node);
    return lFalse;
  } // Can not remove node with this algorithm


  IF_NOT_LOCK(lSiblingIndex) {
    RELEASE_LOCK(_node);
    return lFalse;
  }
  uint16_t lSibling = _bvh.patchNode(lSiblingIndex, PINDEX_SIBLING);

  IF_NOT_LOCK(lParentIndex) {
    RELEASE_LOCK(_node);
    RELEASE_LOCK(lSiblingIndex);
    return lFalse;
  }
  uint16_t lParent           = _bvh.patchNode(lParentIndex, PINDEX_PARENT);
  uint32_t lGrandParentIndex = _bvh.patch_parent(lParent);

  IF_NOT_LOCK(lGrandParentIndex) {
    RELEASE_LOCK(_node);
    RELEASE_LOCK(lSiblingIndex);
    RELEASE_LOCK(lParentIndex);
    return lFalse;
  }
  uint16_t lGrandParent = _bvh.patchNode(lGrandParentIndex, PINDEX_GRAND_PARENT);

  IF_NOT_LOCK(lLeftIndex) {
    RELEASE_LOCK(_node);
    RELEASE_LOCK(lSiblingIndex);
    RELEASE_LOCK(lParentIndex);
    RELEASE_LOCK(lGrandParentIndex);
    return lFalse;
  }

  IF_NOT_LOCK(lRightIndex) {
    RELEASE_LOCK(_node);
    RELEASE_LOCK(lSiblingIndex);
    RELEASE_LOCK(lParentIndex);
    RELEASE_LOCK(lGrandParentIndex);
    RELEASE_LOCK(lLeftIndex);
    return lFalse;
  }

  // FREE LIST:   lNode, lParent
  // INSERT LIST: lLeft, lRight

  // Remove nodes
  if (_bvh.patch_isLeftChild(lParent)) {
    _bvh.patch_left(lGrandParent) = lSiblingIndex;
    _bvh.patch_isLeft(lSibling)   = TRUE;
    _bvh.patch_parent(lSibling)   = lGrandParentIndex;
  } else {
    _bvh.patch_right(lGrandParent) = lSiblingIndex;
    _bvh.patch_isLeft(lSibling)    = FALSE;
    _bvh.patch_parent(lSibling)    = lGrandParentIndex;
  }

  // update Bounding Boxes (temporary)
  _bvh.patchAABBFrom(lGrandParentIndex);

  if (_bvh.orig_surfaceArea(lLeftIndex) > _bvh.orig_surfaceArea(lRightIndex)) {
    return {true, {lLeftIndex, lRightIndex}, {_node, lParentIndex}, {lGrandParentIndex, lSiblingIndex}};
  } else {
    return {true, {lRightIndex, lLeftIndex}, {_node, lParentIndex}, {lGrandParentIndex, lSiblingIndex}};
  }
}


Bittner13Par::INS_RES Bittner13Par::reinsert(
    uint32_t _node, uint32_t _unused, BVHPatch &_bvh, bool _update, SumMin *_sumMin) {

  auto [lBestIndex, lLevelOfBest] = vAltFindNode ? findNode2(_node, _bvh) : findNode1(_node, _bvh);
  if (lBestIndex == _bvh.root()) { return {false, 0, 0}; }

  uint32_t lBestPatchIndex = _bvh.patchIndex(lBestIndex); // Check if node is already patched
  uint16_t lBest;

  if (lBestPatchIndex == UINT32_MAX) {
    // Node is not patched ==> try to lock it
    IF_NOT_LOCK(lBestIndex) { return {false, 0, 0}; }
    lBest = _bvh.patchNode(lBestIndex, _update ? PINDEX_1ST_BEST : PINDEX_2ND_BEST);
  } else if (lBestPatchIndex == PINDEX_GRAND_PARENT) {
    lBest = PINDEX_GRAND_PARENT;
  } else {
    // Node is already owned by this thread ==> no need to lock it -- but move to "correct" patch index
    lBest = _bvh.movePatch(lBestPatchIndex, _update ? PINDEX_1ST_BEST : PINDEX_2ND_BEST);
  }

  uint16_t lNode           = _bvh.patchNode(_node, _update ? PINDEX_1ST_INSERT : PINDEX_2ND_INSERT);
  uint16_t lUnused         = _update ? PINDEX_NODE : PINDEX_PARENT;
  uint32_t lRootIndex      = _bvh.patch_parent(lBest);
  uint32_t lRootPatchIndex = _bvh.patchIndex(lRootIndex);
  uint16_t lRoot;

  if (lRootPatchIndex == UINT32_MAX) {
    IF_NOT_LOCK(lRootIndex) {
      RELEASE_LOCK(lBestIndex);
      return {false, 0, 0};
    }
    lRoot = _bvh.patchNode(lRootIndex, _update ? PINDEX_1ST_ROOT : PINDEX_2ND_ROOT);
  } else {
    lRoot = _bvh.movePatch(lRootPatchIndex, _update ? PINDEX_1ST_ROOT : PINDEX_2ND_ROOT);
  }

  // Insert the unused node
  if (_bvh.isLeftChild(lBest)) {
    _bvh.patch_left(lRoot) = _unused;
    _bvh.isLeft(lUnused)   = TRUE;
  } else {
    _bvh.patch_right(lRoot) = _unused;
    _bvh.isLeft(lUnused)    = FALSE;
  }


  // Insert the other nodes
  _bvh.patch_parent(lUnused) = lRootIndex;
  _bvh.patch_left(lUnused)   = lBestIndex;
  _bvh.patch_right(lUnused)  = _node;

  _bvh.patch_parent(lBest) = _unused;
  _bvh.patch_isLeft(lBest) = TRUE;
  _bvh.patch_parent(lNode) = _unused;
  _bvh.patch_isLeft(lNode) = FALSE;

  if (_update) { _bvh.patchAABBFrom(_unused); }

  return {true, lBestIndex, lRootIndex};
}

void Bittner13Par::fixTree(uint32_t _node, BVH &_bvh, SumMin *_sumMin) {
  uint32_t lNode = _node;

  SPINN_LOCK(lNode);
  uint32_t lLast = _bvh.left(lNode);

  SPINN_LOCK(lLast);
  bool     lLastWasLeft = true;
  uint32_t lCurrSibling = 0;

  AABB  lBBox = _bvh.bbox(lLast);
  float lSum  = SUM_OF(lLast);
  float lMin  = MIN_OF(lLast);
  float lNum  = _bvh.numChildren(lLast);

  float lSArea;
  RELEASE_LOCK(lLast);

  while (true) {
    lCurrSibling = lLastWasLeft ? _bvh.right(lNode) : _bvh.left(lNode);
    SPINN_LOCK(lCurrSibling);

    lBBox.mergeWith(_bvh.bbox(lCurrSibling));
    lSArea                  = lBBox.surfaceArea();
    _bvh.bbox(lNode)        = lBBox;
    _bvh.surfaceArea(lNode) = lSArea;

    lSum                    = lSum + SUM_OF(lCurrSibling) + lSArea;
    lMin                    = min(lMin, MIN_OF(lCurrSibling));
    lNum                    = lNum + _bvh.numChildren(lCurrSibling) + 2;
    SUM_OF(lNode)           = lSum;
    MIN_OF(lNode)           = lMin;
    _bvh.numChildren(lNode) = lNum;

    RELEASE_LOCK(lCurrSibling);

    if (lNode == _bvh.root()) { break; } // We processed the root ==> everything is done

    lLastWasLeft = _bvh.isLeftChild(lNode);
    lLast        = lNode;
    lNode        = _bvh.parent(lNode);

    RELEASE_LOCK(lLast);
    SPINN_LOCK(lNode);
  }

  RELEASE_LOCK(lNode);
}

void Bittner13Par::initSumAndMin(BVH &_bvh, SumMin *_sumMin) {
  if (_bvh.empty()) { return; }

  __uint128_t lBitStack = 0;
  uint32_t    lNode     = _bvh.root();
  uint32_t    lRoot     = lNode;

  while (true) {
    while (!_bvh.isLeaf(lNode)) {
      lBitStack <<= 1;
      lBitStack |= 1;
      lNode = _bvh.left(lNode);
    }

    // Leaf
    SUM_OF(lNode) = _bvh.surfaceArea(lNode);
    MIN_OF(lNode) = _bvh.surfaceArea(lNode);

    // Backtrack if left and right children are processed
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0 && lNode == lRoot) { return; } // We are done
      lNode = _bvh.parent(lNode);

      uint32_t lLeft  = _bvh.left(lNode);
      uint32_t lRight = _bvh.right(lNode);
      AABB     lBBox  = _bvh.bbox(lLeft);
      lBBox.mergeWith(_bvh.bbox(lRight));
      float lSArea            = lBBox.surfaceArea();
      _bvh.bbox(lNode)        = lBBox;
      _bvh.surfaceArea(lNode) = lSArea;
      _bvh.numChildren(lNode) = _bvh.numChildren(lLeft) + _bvh.numChildren(lRight) + 2;
      SUM_OF(lNode)           = SUM_OF(lLeft) + SUM_OF(lRight) + lSArea;
      MIN_OF(lNode)           = min(MIN_OF(lLeft), MIN_OF(lRight));
      lBitStack >>= 1;
    }

    lNode = _bvh.right(_bvh.parent(lNode));
    lBitStack ^= 1;
  }
}




ErrorCode Bittner13Par::runImpl(State &_state) {
  typedef pair<uint32_t, float> TUP;

  unique_ptr<SumMin[]> lSumMin(new SumMin[_state.bvh.size()]);
  SumMin *             _sumMin = lSumMin.get();
  initSumAndMin(_state.bvh, _sumMin);

  uint32_t lNumNodes  = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  uint32_t lChunkSize = lNumNodes / vNumChunks;
  lNumNodes           = lChunkSize * vNumChunks;
  auto lComp          = [](TUP const &_l, TUP const &_r) -> bool { return _l.second > _r.second; };

  vector<TUP>      lTodoList;
  vector<BVHPatch> lPatches;
  vector<bool>     lSkipp;
  vector<uint32_t> lFixList;
  lTodoList.resize(_state.bvh.size());
  lPatches.resize(lChunkSize, BVHPatch(&_state.bvh));
  lSkipp.resize(lChunkSize);
  lFixList.resize(lChunkSize * 3);

  uint32_t lSkipped = 0;

  // Init List
#pragma omp parallel for
  for (uint32_t i = 0; i < lTodoList.size(); ++i) {
    lTodoList[i] = {i, 0.0f};
    RELEASE_LOCK(i);
  }

  random_device                      lRD;
  mt19937                            lPRNG(lRD());
  uniform_int_distribution<uint32_t> lDis(0, lTodoList.size() - 1);


  /*****  ___  ___      _         _                         *****/
  /*****  |  \/  |     (_)       | |                        *****/
  /*****  | .  . | __ _ _ _ __   | |     ___   ___  _ __    *****/
  /*****  | |\/| |/ _` | | '_ \  | |    / _ \ / _ \| '_ \   *****/
  /*****  | |  | | (_| | | | | | | |___| (_) | (_) | |_) |  *****/
  /*****  \_|  |_/\__,_|_|_| |_| \_____/\___/ \___/| .__/   *****/
  /*****                                           | |      *****/
  /*****                                           |_|      *****/


  for (uint32_t i = 0; i < vMaxNumStepps; ++i) {
#if ENABLE_PROGRESS_BAR
    progress(fmt::format("Stepp {:<3}; SAH: {:<6.5}", i, _state.bvh.calcSAH()), i, vMaxNumStepps - 1);
#endif

    /*   _____ _                     __      _____      _           _     _   _           _             */
    /*  /  ___| |                   /  | _  /  ___|    | |         | |   | \ | |         | |            */
    /*  \ `--.| |_ ___ _ __  _ __   `| |(_) \ `--.  ___| | ___  ___| |_  |  \| | ___   __| | ___  ___   */
    /*   `--. \ __/ _ \ '_ \| '_ \   | |     `--. \/ _ \ |/ _ \/ __| __| | . ` |/ _ \ / _` |/ _ \/ __|  */
    /*  /\__/ / ||  __/ |_) | |_) | _| |__  /\__/ /  __/ |  __/ (__| |_  | |\  | (_) | (_| |  __/\__ \  */
    /*  \____/ \__\___| .__/| .__/  \___(_) \____/ \___|_|\___|\___|\__| \_| \_/\___/ \__,_|\___||___/  */
    /*                | |   | |                                                                         */
    /*                |_|   |_|                                                                         */

    if (vRandom) {
      // === Random suffle ===
      shuffle(begin(lTodoList), end(lTodoList), lPRNG);

    } else {
      // === Metric selction ===
#pragma omp parallel for
      for (uint32_t j = 0; j < _state.bvh.size(); ++j) {
        float lSA = _state.bvh.surfaceArea(j);

        bool lIsRoot       = j == _state.bvh.root();
        bool lParentIsRoot = _state.bvh.parent(j) == _state.bvh.root();
        bool lCanRM        = !lIsRoot && !lParentIsRoot && !_state.bvh.isLeaf(j);

        float lCost  = lCanRM ? ((lSA * lSA * lSA * (float)_state.bvh.numChildren(j)) / (SUM_OF(j) * MIN_OF(j))) : 0.0f;
        lTodoList[j] = {j, lCost};
      }


      nth_element(begin(lTodoList), begin(lTodoList) + lNumNodes, end(lTodoList), lComp);
      if (vSortBatch) { sort(begin(lTodoList), begin(lTodoList) + lNumNodes, lComp); }
      if (vShuffleList) { shuffle(begin(lTodoList), begin(lTodoList) + lNumNodes, lPRNG); }
    }

    // Separate the batch into chunks
    for (uint32_t j = 0; j < vNumChunks; ++j) {


      /*   _____ _                     _____     ______     _                     _     _   _           _             */
      /*  /  ___| |                   / __  \ _  | ___ \   (_)                   | |   | \ | |         | |            */
      /*  \ `--.| |_ ___ _ __  _ __   `' / /'(_) | |_/ /___ _ _ __  ___  ___ _ __| |_  |  \| | ___   __| | ___  ___   */
      /*   `--. \ __/ _ \ '_ \| '_ \    / /      |    // _ \ | '_ \/ __|/ _ \ '__| __| | . ` |/ _ \ / _` |/ _ \/ __|  */
      /*  /\__/ / ||  __/ |_) | |_) | ./ /___ _  | |\ \  __/ | | | \__ \  __/ |  | |_  | |\  | (_) | (_| |  __/\__ \  */
      /*  \____/ \__\___| .__/| .__/  \_____/(_) \_| \_\___|_|_| |_|___/\___|_|   \__| \_| \_/\___/ \__,_|\___||___/  */
      /*                | |   | |                                                                                     */
      /*                |_|   |_|                                                                                     */

#pragma omp parallel for schedule(dynamic, 128)
      for (uint32_t k = 0; k < lChunkSize; ++k) {
        lPatches[k].clear();

        auto [lNodeIndex, _]                  = lTodoList[vOffsetAccess ? k * vNumChunks + j : j * lChunkSize + k];
        auto [lRes, lTInsert, lTUnused, lETC] = removeNode(lNodeIndex, lPatches[k], _sumMin);
        auto [l1stIndex, l2ndIndex]           = lTInsert;
        auto [lU1, lU2]                       = lTUnused;
        auto [lGP, lSIB]                      = lETC;

        if (!lRes) {
          lSkipp[k] = true;
          continue;
        }

        INS_RES lR1 = reinsert(l1stIndex, lU1, lPatches[k], true, _sumMin);
        INS_RES lR2 = reinsert(l2ndIndex, lU2, lPatches[k], false, _sumMin);
        if (!lR1.res || !lR2.res) {
          lSkipp[k] = true;

          // Unlock Nodes
          RELEASE_LOCK(l1stIndex);
          RELEASE_LOCK(l2ndIndex);
          RELEASE_LOCK(lU1);
          RELEASE_LOCK(lU2);
          RELEASE_LOCK(lGP);
          RELEASE_LOCK(lSIB);
          if (lR1.res) {
            RELEASE_LOCK(lR1.best);
            RELEASE_LOCK(lR1.root);
          }
          if (lR2.res) {
            RELEASE_LOCK(lR2.best);
            RELEASE_LOCK(lR2.root);
          }
          continue;
        }

        lSkipp[k]           = false;
        lFixList[k * 3 + 0] = lGP;
        lFixList[k * 3 + 1] = lU1;
        lFixList[k * 3 + 2] = lU2;
      }


      /*   _____ _                     _____      ___              _        ______     _       _                 */
      /*  /  ___| |                   |____ |_   / _ \            | |       | ___ \   | |     | |                */
      /*  \ `--.| |_ ___ _ __  _ __       / (_) / /_\ \_ __  _ __ | |_   _  | |_/ /_ _| |_ ___| |__   ___  ___   */
      /*   `--. \ __/ _ \ '_ \| '_ \      \ \   |  _  | '_ \| '_ \| | | | | |  __/ _` | __/ __| '_ \ / _ \/ __|  */
      /*  /\__/ / ||  __/ |_) | |_) | .___/ /_  | | | | |_) | |_) | | |_| | | | | (_| | || (__| | | |  __/\__ \  */
      /*  \____/ \__\___| .__/| .__/  \____/(_) \_| |_/ .__/| .__/|_|\__, | \_|  \__,_|\__\___|_| |_|\___||___/  */
      /*                | |   | |                     | |   | |       __/ |                                      */
      /*                |_|   |_|                     |_|   |_|      |___/                                       */

#pragma omp parallel for
      for (uint32_t k = 0; k < lChunkSize; ++k) {
        if (lSkipp[k]) { continue; }

        // Reset locks
        for (uint32_t l = 0; l < NNode; ++l) {
          uint32_t lIDX = lPatches[k].getPatchedNodeIndex(l);
          if (lIDX == UINT32_MAX) { continue; }
          RELEASE_LOCK(lIDX);
        }

        lPatches[k].apply();
      }

      /*   _____ _                       ___    ______ _        _   _            _                   */
      /*  /  ___| |                     /   |_  |  ___(_)      | | | |          | |                  */
      /*  \ `--.| |_ ___ _ __  _ __    / /| (_) | |_   ___  __ | |_| |__   ___  | |_ _ __ ___  ___   */
      /*   `--. \ __/ _ \ '_ \| '_ \  / /_| |   |  _| | \ \/ / | __| '_ \ / _ \ | __| '__/ _ \/ _ \  */
      /*  /\__/ / ||  __/ |_) | |_) | \___  |_  | |   | |>  <  | |_| | | |  __/ | |_| | |  __/  __/  */
      /*  \____/ \__\___| .__/| .__/      |_(_) \_|   |_/_/\_\  \__|_| |_|\___|  \__|_|  \___|\___|  */
      /*                | |   | |                                                                    */
      /*                |_|   |_|                                                                    */


#pragma omp parallel for reduction(+ : lSkipped) schedule(dynamic, 128)
      for (uint32_t k = 0; k < lChunkSize; ++k) {
        if (lSkipp[k]) {
          lSkipped++;
          continue;
        }

        fixTree(lFixList[k * 3 + 0], _state.bvh, _sumMin);
        fixTree(lFixList[k * 3 + 1], _state.bvh, _sumMin);
        fixTree(lFixList[k * 3 + 2], _state.bvh, _sumMin);
      }
    }
  }

  PROGRESS_DONE;

  getLogger()->info("Skipped {:<8} of {:<8} -- {}%",
                    lSkipped,
                    lNumNodes * vMaxNumStepps,
                    (int)(((float)lSkipped / (float)(lNumNodes * vMaxNumStepps)) * 100));

  _state.bvh.fixLevels();
  return ErrorCode::OK;
}
