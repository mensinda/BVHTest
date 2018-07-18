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


uint32_t Bittner13Par::findNode1(uint32_t _n, BVHPatch &_bvh) {
  float          lBestCost      = HUGE_VALF;
  uint32_t       lBestNodeIndex = 0;
  BVHNode const *lNode          = _bvh.getOrig(_n);
  AABB const &   lNodeBBox      = lNode->bbox;
  float          lSArea         = lNode->surfaceArea;
  uint32_t       lSize          = 1;
  HelperStruct   lPQ[QUEUE_SIZE];
  HelperStruct * lBegin = lPQ;

  lPQ[0] = {_bvh.root(), 0.0f, 0};
  while (lSize > 0) {
    HelperStruct lCurr     = lPQ[0];
    BVHNodePatch lCurrNode = _bvh.getSubset(lCurr.node);
    AABB         lBBox     = _bvh.getAABB(lCurr.node);
    CUDA_pop_heap(lBegin, lBegin + lSize);
    lSize--;

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    float lNewInduced = -1 * lBBox.surfaceArea();

    lBBox.mergeWith(lNodeBBox);
    float lDirectCost = lBBox.surfaceArea();
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = lCurr.node;
    }

    lNewInduced += lTotalCost;
    if ((lNewInduced + lSArea) < lBestCost) {
      if (!lCurrNode.isLeaf()) {
        assert(lSize + 2 < QUEUE_SIZE);
        lPQ[lSize + 0] = {lCurrNode.left, lNewInduced, lCurr.level + 1};
        lPQ[lSize + 1] = {lCurrNode.right, lNewInduced, lCurr.level + 1};
        CUDA_push_heap(lBegin, lBegin + lSize + 1);
        CUDA_push_heap(lBegin, lBegin + lSize + 2);
        lSize += 2;
      }
    }
  }

  return lBestNodeIndex;
}

uint32_t Bittner13Par::findNode2(uint32_t _n, BVHPatch &_bvh) {
  float          lBestCost      = HUGE_VALF;
  uint32_t       lBestNodeIndex = 0;
  BVHNode const *lNode          = _bvh.getOrig(_n);
  AABB const &   lNodeBBox      = lNode->bbox;
  float          lSArea         = lNode->surfaceArea;
  float          lMin           = 0.0f;
  float          lMax           = HUGE_VALF;
  uint32_t       lMinIndex      = 0;
  uint32_t       lMaxIndex      = 1;
  HelperStruct   lPQ[vAltFNQSize];
  HelperStruct   lCurr;

  // Init
  for (uint32_t i = 0; i < vAltFNQSize; ++i) { lPQ[i].cost = HUGE_VALF; }

  lPQ[0] = {_bvh.root(), 0.0f, 0};
  while (lMin < HUGE_VALF) {
    lCurr                  = lPQ[lMinIndex];
    lPQ[lMinIndex].cost    = HUGE_VALF;
    BVHNodePatch lCurrNode = _bvh.getSubset(lCurr.node);
    auto         lBBox     = _bvh.getAABB(lCurr.node);

    if ((lCurr.cost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    float lNewInduced = -1 * lBBox.surfaceArea();

    lBBox.mergeWith(lNodeBBox);
    float lDirectCost = lBBox.surfaceArea();
    float lTotalCost  = lCurr.cost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = lCurr.node;
    }

    lNewInduced += lTotalCost;
    if ((lNewInduced + lSArea) < lBestCost && !lCurrNode.isLeaf()) {
      lPQ[lMinIndex] = {lCurrNode.left, lNewInduced, lCurr.level + 1};
      lPQ[lMaxIndex] = {lCurrNode.right, lNewInduced, lCurr.level + 1};
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

#define SPINN_LOCK(N, VAL)                                                                                             \
  {                                                                                                                    \
    uint32_t __lock = 0;                                                                                               \
    while (_sumMin[N].flag.compare_exchange_strong(__lock, VAL)) {}                                                    \
  }

#define IF_NOT_LOCK(N, VAL)                                                                                            \
  lLock = 0;                                                                                                           \
  if (!_sumMin[N].flag.compare_exchange_strong(lLock, VAL))

#define RELEASE_LOCK_S(N, VAL)                                                                                         \
  {                                                                                                                    \
    uint32_t __lock = VAL;                                                                                             \
    _sumMin[N].flag.compare_exchange_strong(__lock, 0);                                                                \
  }

#define RELEASE_LOCK(N) _sumMin[N].flag.store(0);

Bittner13Par::RM_RES Bittner13Par::removeNode(uint32_t _node, BVHPatch &_bvh, SumMin *) {
  RM_RES lFalse = {false, {0, 0}, {0, 0}, {0, 0}};
  if (_bvh.getOrig(_node)->isLeaf() || _node == _bvh.root()) { return lFalse; }

  BVHNodePatch *lNode         = _bvh.patchNode(_node, PINDEX_NODE);
  uint32_t      lSiblingIndex = _bvh.sibling(*lNode);
  uint32_t      lParentIndex  = lNode->parent;

  if (lParentIndex == _bvh.root()) { return lFalse; } // Can not remove node with this algorithm

  BVHNodePatch *lSibling          = _bvh.patchNode(lSiblingIndex, PINDEX_SIBLING);
  BVHNodePatch *lParent           = _bvh.patchNode(lParentIndex, PINDEX_PARENT);
  uint32_t      lGrandParentIndex = lParent->parent;
  BVHNodePatch *lGrandParent      = _bvh.patchNode(lGrandParentIndex, PINDEX_GRAND_PARENT);


  // FREE LIST:   lNode, lParent
  // INSERT LIST: lLeft, lRight

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

  // update Bounding Boxes (temporary)
  _bvh.patchAABBFrom(lGrandParentIndex, 0);

  if (_bvh.getOrig(lNode->left)->surfaceArea > _bvh.getOrig(lNode->right)->surfaceArea) {
    return {true, {lNode->left, lNode->right}, {_node, lParentIndex}, {lGrandParentIndex, lSiblingIndex}};
  } else {
    return {true, {lNode->right, lNode->left}, {_node, lParentIndex}, {lGrandParentIndex, lSiblingIndex}};
  }
}


Bittner13Par::INS_RES Bittner13Par::reinsert(uint32_t _node, uint32_t _unused, BVHPatch &_bvh, bool _update, SumMin *) {
  uint32_t lBestIndex = vAltFindNode ? findNode2(_node, _bvh) : findNode1(_node, _bvh);
  if (lBestIndex == _bvh.root()) { return {false, 0, 0}; }

  uint32_t      lBestPatchIndex = _bvh.patchIndex(lBestIndex); // Check if node is already patched
  BVHNodePatch *lBest           = nullptr;

  if (lBestPatchIndex == UINT32_MAX) {
    lBest = _bvh.patchNode(lBestIndex, _update ? PINDEX_1ST_BEST : PINDEX_2ND_BEST);
  } else if (lBestPatchIndex == PINDEX_GRAND_PARENT) {
    lBest = _bvh.getPatchedNode(PINDEX_GRAND_PARENT);
  } else {
    lBest = _bvh.movePatch(lBestPatchIndex, _update ? PINDEX_1ST_BEST : PINDEX_2ND_BEST);
  }

  BVHNodePatch *lNode           = _bvh.patchNode(_node, _update ? PINDEX_1ST_INSERT : PINDEX_2ND_INSERT);
  BVHNodePatch *lUnused         = _bvh.getPatchedNode(_update ? PINDEX_NODE : PINDEX_PARENT);
  uint32_t      lRootIndex      = lBest->parent;
  uint32_t      lRootPatchIndex = _bvh.patchIndex(lRootIndex);
  BVHNodePatch *lRoot           = nullptr;

  if (lRootPatchIndex == UINT32_MAX) {
    lRoot = _bvh.patchNode(lRootIndex, _update ? PINDEX_1ST_ROOT : PINDEX_2ND_ROOT);
  } else {
    lRoot = _bvh.movePatch(lRootPatchIndex, _update ? PINDEX_1ST_ROOT : PINDEX_2ND_ROOT);
  }

  // Insert the unused node
  if (lBest->isLeftChild()) {
    lRoot->left     = _unused;
    lUnused->isLeft = TRUE;
  } else {
    lRoot->right    = _unused;
    lUnused->isLeft = FALSE;
  }


  // Insert the other nodes
  lUnused->parent = lRootIndex;
  lUnused->left   = lBestIndex;
  lUnused->right  = _node;

  lBest->parent = _unused;
  lBest->isLeft = TRUE;
  lNode->parent = _unused;
  lNode->isLeft = FALSE;

  if (_update) { _bvh.patchAABBFrom(_unused, 1); }

  return {true, lBestIndex, lRootIndex};
}

void Bittner13Par::fixTree(uint32_t _node, BVH &_bvh, SumMin *_sumMin) {
  uint32_t lNode = _node;
  _node++; // Make sure _node is never 0

  SPINN_LOCK(lNode, _node);
  uint32_t lLastIndex = NODE->left;

  SPINN_LOCK(lLastIndex, _node);
  BVHNode *lLast             = _bvh[lLastIndex];
  bool     lLastWasLeft      = true;
  uint32_t lCurrSiblingIndex = 0;
  BVHNode *lCurrSibling      = nullptr;

  AABB  lBBox = LEFT->bbox;
  float lSum  = SUM_OF(lLastIndex);
  float lMin  = MIN_OF(lLastIndex);
  float lNum  = lLast->numChildren;

  float lSArea;
  RELEASE_LOCK_S(lLastIndex, _node);

  while (true) {
    lCurrSiblingIndex = lLastWasLeft ? NODE->right : NODE->left;
    SPINN_LOCK(lCurrSiblingIndex, _node);

    lCurrSibling = _bvh[lCurrSiblingIndex];

    lBBox.mergeWith(lCurrSibling->bbox);
    lSArea            = lBBox.surfaceArea();
    NODE->bbox        = lBBox;
    NODE->surfaceArea = lSArea;

    lSum              = lSum + SUM_OF(lCurrSiblingIndex) + lSArea;
    lMin              = min(lMin, MIN_OF(lCurrSiblingIndex));
    lNum              = lNum + lCurrSibling->numChildren + 2;
    SUM_OF(lNode)     = lSum;
    MIN_OF(lNode)     = lMin;
    NODE->numChildren = lNum;

    RELEASE_LOCK_S(lCurrSiblingIndex, _node);

    if (lNode == _bvh.root()) { break; } // We processed the root ==> everything is done

    lLastWasLeft = NODE->isLeftChild();
    lLastIndex   = lNode;
    lLast        = _bvh[lLastIndex];
    lNode        = NODE->parent;

    RELEASE_LOCK_S(lLastIndex, _node);
    SPINN_LOCK(lNode, _node);
  }

  RELEASE_LOCK_S(lNode, _node);
}

void Bittner13Par::initSumAndMin(BVH &_bvh, SumMin *_sumMin) {
  if (_bvh.empty()) { return; }

  __uint128_t lBitStack = 0;
  uint32_t    lNode     = _bvh.root();
  uint32_t    lRoot     = lNode;

  while (true) {
    while (!NODE->isLeaf()) {
      lBitStack <<= 1;
      lBitStack |= 1;
      lNode = NODE->left;
    }

    // Leaf
    SUM_OF(lNode) = NODE->surfaceArea;
    MIN_OF(lNode) = NODE->surfaceArea;

    // Backtrack if left and right children are processed
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0 && lNode == lRoot) { return; } // We are done
      lNode = NODE->parent;

      AABB lBBox = LEFT->bbox;
      lBBox.mergeWith(RIGHT->bbox);
      float lSArea      = lBBox.surfaceArea();
      NODE->bbox        = lBBox;
      NODE->surfaceArea = lSArea;
      NODE->numChildren = LEFT->numChildren + RIGHT->numChildren + 2;
      SUM_OF(lNode)     = SUM_OF(NODE->left) + SUM_OF(NODE->right) + lSArea;
      MIN_OF(lNode)     = min(MIN_OF(NODE->left), MIN_OF(NODE->right));
      lBitStack >>= 1;
    }

    lNode = PARENT->right;
    lBitStack ^= 1;
  }
}


ErrorCode Bittner13Par::setup(State &_state) {
  uint32_t lNumNodes  = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  uint32_t lChunkSize = lNumNodes / vNumChunks;

  vTodoList.resize(_state.bvh.size());
  vPatches.resize(lChunkSize, BVHPatch(&_state.bvh));
  vSkipp.resize(lChunkSize);
  vFixList.resize(lChunkSize * 3);
  vSumMin         = unique_ptr<SumMin[]>(new SumMin[_state.bvh.size()]);
  SumMin *_sumMin = vSumMin.get();

  // Init List
#pragma omp parallel for
  for (uint32_t i = 0; i < vTodoList.size(); ++i) {
    vTodoList[i] = {i, 0.0f};
    RELEASE_LOCK(i);
  }

  initSumAndMin(_state.bvh, _sumMin);
  return ErrorCode::OK;
}

ErrorCode Bittner13Par::runImpl(State &_state) {
  uint32_t lNumNodes  = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  uint32_t lChunkSize = lNumNodes / vNumChunks;
  lNumNodes           = lChunkSize * vNumChunks;
  SumMin *_sumMin     = vSumMin.get();
  auto    lComp       = [](TUP const &_l, TUP const &_r) -> bool { return _l.second > _r.second; };

  uint32_t lSkipped = 0;

  random_device                      lRD;
  mt19937                            lPRNG(lRD());
  uniform_int_distribution<uint32_t> lDis(0, vTodoList.size() - 1);


  /*****  ___  ___      _         _                         *****/
  /*****  |  \/  |     (_)       | |                        *****/
  /*****  | .  . | __ _ _ _ __   | |     ___   ___  _ __    *****/
  /*****  | |\/| |/ _` | | '_ \  | |    / _ \ / _ \| '_ \   *****/
  /*****  | |  | | (_| | | | | | | |___| (_) | (_) | |_) |  *****/
  /*****  \_|  |_/\__,_|_|_| |_| \_____/\___/ \___/| .__/   *****/
  /*****                                           | |      *****/
  /*****                                           |_|      *****/

  benchmarkInitData(_state, [&]() { return _state.bvh.calcSAH(); });

  for (uint32_t i = 0; i < vMaxNumStepps; ++i) {
#if ENABLE_PROGRESS_BAR
    progress(fmt::format("Stepp {:<3}; SAH: {:<6.5}", i, _state.bvh.calcSAH()), i, vMaxNumStepps - 1);
#endif

    benchmarkStartTimer(_state);

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
      shuffle(begin(vTodoList), end(vTodoList), lPRNG);

    } else {
      // === Metric selction ===
#pragma omp parallel for
      for (uint32_t j = 0; j < _state.bvh.size(); ++j) {
        BVHNode const *lNode = _state.bvh[j];
        float          lSA   = lNode->surfaceArea;

        float lCost  = (lSA * lSA * lSA * (float)lNode->numChildren) / (SUM_OF(j) * MIN_OF(j));
        vTodoList[j] = {j, lCost};
      }


      nth_element(begin(vTodoList), begin(vTodoList) + lNumNodes, end(vTodoList), lComp);
      if (vSortBatch) { sort(begin(vTodoList), begin(vTodoList) + lNumNodes, lComp); }
      if (vShuffleList) { shuffle(begin(vTodoList), begin(vTodoList) + lNumNodes, lPRNG); }
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
        vPatches[k].clear();

        auto [lNodeIndex, _]                  = vTodoList[vOffsetAccess ? k * vNumChunks + j : j * lChunkSize + k];
        auto [lRes, lTInsert, lTUnused, lETC] = removeNode(lNodeIndex, vPatches[k], _sumMin);
        auto [l1stIndex, l2ndIndex]           = lTInsert;
        auto [lU1, lU2]                       = lTUnused;
        auto [lGP, lSIB]                      = lETC;

        if (!lRes) {
          vSkipp[k] = true;
          continue;
        }

        INS_RES lR1 = reinsert(l1stIndex, lU1, vPatches[k], true, _sumMin);
        INS_RES lR2 = reinsert(l2ndIndex, lU2, vPatches[k], false, _sumMin);
        if (!lR1.res || !lR2.res) {
          vSkipp[k] = true;
          continue;
        }

        vSkipp[k]           = false;
        vFixList[k * 3 + 0] = lGP;
        vFixList[k * 3 + 1] = lU1;
        vFixList[k * 3 + 2] = lU2;
      }


      /*   _____ _                     _____      ___              _        ______     _       _                 */
      /*  /  ___| |                   |____ |_   / _ \            | |       | ___ \   | |     | |                */
      /*  \ `--.| |_ ___ _ __  _ __       / (_) / /_\ \_ __  _ __ | |_   _  | |_/ /_ _| |_ ___| |__   ___  ___   */
      /*   `--. \ __/ _ \ '_ \| '_ \      \ \   |  _  | '_ \| '_ \| | | | | |  __/ _` | __/ __| '_ \ / _ \/ __|  */
      /*  /\__/ / ||  __/ |_) | |_) | .___/ /_  | | | | |_) | |_) | | |_| | | | | (_| | || (__| | | |  __/\__ \  */
      /*  \____/ \__\___| .__/| .__/  \____/(_) \_| |_/ .__/| .__/|_|\__, | \_|  \__,_|\__\___|_| |_|\___||___/  */
      /*                | |   | |                     | |   | |       __/ |                                      */
      /*                |_|   |_|                     |_|   |_|      |___/                                       */


      // #pragma omp parallel for
      for (uint32_t k = 0; k < lChunkSize; ++k) {
        if (vSkipp[k]) { continue; }

        uint32_t l = 0;

        for (; l < NNode; ++l) {
          uint32_t lIDX = vPatches[k].getPatchedNodeIndex(l);
          if (lIDX != UINT32_MAX) {
            if (_sumMin[lIDX].flag.fetch_add(1) != 0) { goto FAILED; }
          }
        }

        continue;

      FAILED:
        vSkipp[k] = true;

        for (; l < UINT32_MAX; --l) {
          uint32_t lIDX = vPatches[k].getPatchedNodeIndex(l);
          if (lIDX != UINT32_MAX) { _sumMin[lIDX].flag -= 1; }
        }
      }

#pragma omp parallel for
      for (uint32_t k = 0; k < lChunkSize; ++k) {
        for (uint32_t l = 0; l < NNode; ++l) {
          uint32_t lIDX = vPatches[k].getPatchedNodeIndex(l);
          if (lIDX != UINT32_MAX) { RELEASE_LOCK(lIDX); }
        }

        if (vSkipp[k]) { continue; }

        vPatches[k].apply();
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
        if (vSkipp[k]) {
          lSkipped++;
          continue;
        }

        fixTree(vFixList[k * 3 + 0], _state.bvh, _sumMin);
        fixTree(vFixList[k * 3 + 1], _state.bvh, _sumMin);
        fixTree(vFixList[k * 3 + 2], _state.bvh, _sumMin);
      }
    }

    benchmarkRecordData(_state, [&]() { return _state.bvh.calcSAH(); });
  }

  PROGRESS_DONE;

  getLogger()->info("Skipped {:<8} of {:<8} -- {}%",
                    lSkipped,
                    lNumNodes * vMaxNumStepps,
                    (int)(((float)lSkipped / (float)(lNumNodes * vMaxNumStepps)) * 100));

  _state.optData.numSkipped = lSkipped;
  _state.optData.numTotal   = lNumNodes * vMaxNumStepps;

  _state.bvh.fixLevels();
  return ErrorCode::OK;
}

void Bittner13Par::teardown(State &) { vSumMin = nullptr; }
