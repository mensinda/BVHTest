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
  vNumChunks    = _j.value("numChunks", vNumChunks);
  vBatchPercent = _j.value("batchPercent", vBatchPercent);
  vRandom       = _j.value("random", vRandom);
  vSortBatch    = _j.value("sort", vSortBatch);
  vShuffleList  = _j.value("shuffle", vShuffleList);

  if (vBatchPercent <= 0.01f) vBatchPercent = 0.01f;
  if (vBatchPercent >= 75.0f) vBatchPercent = 75.0f;
}

json Bittner13Par::toJSON() const {
  json lJSON            = OptimizerBase::toJSON();
  lJSON["maxNumStepps"] = vMaxNumStepps;
  lJSON["numChunks"]    = vNumChunks;
  lJSON["batchPercent"] = vBatchPercent;
  lJSON["random"]       = vRandom;
  lJSON["sort"]         = vSortBatch;
  lJSON["shuffle"]      = vShuffleList;
  return lJSON;
}


pair<uint32_t, uint32_t> Bittner13Par::findNodeForReinsertion(uint32_t _n, PATCH &_bvh) {
  typedef tuple<uint32_t, float, uint32_t> T1; // Node Ind, cost, tree level

  float                    lBestCost      = numeric_limits<float>::infinity();
  pair<uint32_t, uint32_t> lBestNodeIndex = {0, 0};
  BVHNode const *          lNode          = _bvh[_n];
  AABB const &             lNodeBBox      = lNode->bbox;
  float                    lSArea         = lNode->surfaceArea;
  auto                     lComp          = [](T1 const &_l, T1 const &_r) -> bool { return get<1>(_l) > get<1>(_r); };
  priority_queue<T1, vector<T1>, decltype(lComp)> lPQ(lComp);

  lPQ.push({_bvh.root(), 0.0f, 0});
  while (!lPQ.empty()) {
    auto [lCurrNodeIndex, lCurrCost, lLevel] = lPQ.top();
    BVHNode *lCurrNode                       = _bvh[lCurrNodeIndex];
    auto [lAABB, lCurrSArea]                 = _bvh.getAABB(lCurrNodeIndex, lLevel);
    lPQ.pop();

    if ((lCurrCost + lSArea) >= lBestCost) {
      // Early termination - not possible to further optimize
      break;
    }

    lAABB.mergeWith(lNodeBBox);
    float lDirectCost = lAABB.surfaceArea();
    float lTotalCost  = lCurrCost + lDirectCost;
    if (lTotalCost < lBestCost) {
      // Merging here improves the total SAH cost
      lBestCost      = lTotalCost;
      lBestNodeIndex = {lCurrNodeIndex, lLevel};
    }

    float lNewInduced = lTotalCost - lCurrSArea;
    if ((lNewInduced + lSArea) < lBestCost) {
      if (!lCurrNode->isLeaf()) {
        lPQ.push({lCurrNode->left, lNewInduced, lLevel + 1});
        lPQ.push({lCurrNode->right, lNewInduced, lLevel + 1});
      }
    }
  }

  return lBestNodeIndex;
}

#define SPINN_LOCK(N)                                                                                                  \
  while (ATO_OF(N).test_and_set(memory_order_acquire)) { this_thread::yield(); }
#define IF_NOT_LOCK(N) if (ATO_OF(N).test_and_set(memory_order_acquire))
#define RELEASE_LOCK(N) ATO_OF(N).clear(memory_order_release);

Bittner13Par::RM_RES Bittner13Par::removeNode(uint32_t _node, PATCH &_bvh, SumMin *_sumMin) {
  RM_RES lFalse = {false, {0, 0}, {0, 0}, 0};
  if (_bvh[_node]->isLeaf() || _node == _bvh.root()) { return lFalse; }

  IF_NOT_LOCK(_node) { return lFalse; }

  BVHNode *lNode         = _bvh.patchNode(_node);
  uint32_t lSiblingIndex = _bvh.sibling(*lNode);
  uint32_t lParentIndex  = lNode->parent;

  IF_NOT_LOCK(lSiblingIndex) {
    ATO_OF(_node).clear(memory_order_release);
    return lFalse;
  }
  BVHNode *lSibling = _bvh.patchNode(lSiblingIndex);

  IF_NOT_LOCK(lParentIndex) {
    ATO_OF(_node).clear(memory_order_release);
    ATO_OF(lSiblingIndex).clear(memory_order_release);
    return lFalse;
  }
  BVHNode *lParent           = _bvh.patchNode(lParentIndex);
  uint32_t lGrandParentIndex = lParent->parent;

  IF_NOT_LOCK(lGrandParentIndex) {
    ATO_OF(_node).clear(memory_order_release);
    ATO_OF(lSiblingIndex).clear(memory_order_release);
    ATO_OF(lParentIndex).clear(memory_order_release);
    return lFalse;
  }
  BVHNode *lGrandParent = _bvh.patchNode(lGrandParentIndex);

  IF_NOT_LOCK(lNode->left) {
    ATO_OF(_node).clear(memory_order_release);
    ATO_OF(lSiblingIndex).clear(memory_order_release);
    ATO_OF(lParentIndex).clear(memory_order_release);
    ATO_OF(lGrandParentIndex).clear(memory_order_release);
    return lFalse;
  }

  IF_NOT_LOCK(lNode->right) {
    ATO_OF(_node).clear(memory_order_release);
    ATO_OF(lSiblingIndex).clear(memory_order_release);
    ATO_OF(lParentIndex).clear(memory_order_release);
    ATO_OF(lGrandParentIndex).clear(memory_order_release);
    ATO_OF(lNode->left).clear(memory_order_release);
    return lFalse;
  }

  BVHNode *lLeft  = _bvh.patchNode(lNode->left);
  BVHNode *lRight = _bvh.patchNode(lNode->right);

  // FREE LIST:   lNode, lParent
  // INSERT LIST: lLeft, lRight

  float lLeftSA  = lLeft->surfaceArea;
  float lRightSA = lRight->surfaceArea;


  if (lParentIndex == _bvh.root()) { return lFalse; } // Can not remove node with this algorithm

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
  _bvh.patchAABBFrom(lGrandParentIndex);

  if (lLeftSA > lRightSA) {
    return {true, {lNode->left, lNode->right}, {_node, lParentIndex}, lGrandParentIndex};
  } else {
    return {true, {lNode->right, lNode->left}, {_node, lParentIndex}, lGrandParentIndex};
  }
}


bool Bittner13Par::reinsert(uint32_t _node, uint32_t _unused, PATCH &_bvh, bool _update, SumMin *_sumMin) {
  auto [lBestIndex, lLevelOfBest] = findNodeForReinsertion(_node, _bvh);
  if (lBestIndex == _bvh.root()) { return false; }

  uint32_t lBestPatchIndex = _bvh.patchIndex(lBestIndex); // Check if node is already patched
  BVHNode *lBest           = nullptr;

  if (lBestPatchIndex == UINT32_MAX) {
    // Node is not patched ==> try to lock it
    IF_NOT_LOCK(lBestIndex) { return false; }
    lBest = _bvh.patchNode(lBestIndex);
  } else {
    // Node is already owned by this thread ==> no need to lock it
    lBest = _bvh.getPatchedNode(lBestPatchIndex);
  }

  BVHNode *lNode           = _bvh[_node];
  BVHNode *lUnused         = _bvh[_unused];
  uint32_t lRootIndex      = lBest->parent;
  uint32_t lRootPatchIndex = _bvh.patchIndex(lRootIndex);
  BVHNode *lRoot           = nullptr;

  if (lRootPatchIndex == UINT32_MAX) {
    IF_NOT_LOCK(lRootIndex) {
      ATO_OF(lBestIndex).clear(memory_order_release);
      return false;
    }
    lRoot = _bvh.patchNode(lRootIndex);
  } else {
    lRoot = _bvh.getPatchedNode(lRootPatchIndex);
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

  if (_update) {
    _bvh.nodeUpdated(lBestIndex, lLevelOfBest);
    _bvh.patchAABBFrom(_unused);
  }

  return true;
}

void Bittner13Par::fixTree(uint32_t _node, BVH &_bvh, SumMin *_sumMin) {
  uint32_t lNode = _node;

  SPINN_LOCK(lNode);
  uint32_t lLastIndex = NODE.left;

  SPINN_LOCK(lLastIndex);
  BVHNode *lLast             = &_bvh[lLastIndex];
  bool     lLastWasLeft      = true;
  uint32_t lCurrSiblingIndex = 0;
  BVHNode *lCurrSibling      = nullptr;

  AABB  lBBox = LEFT.bbox;
  float lSum  = SUM_OF(lLastIndex);
  float lMin  = MIN_OF(lLastIndex);
  float lNum  = lLast->numChildren;

  float lSArea;
  RELEASE_LOCK(lLastIndex);

  while (true) {
    lCurrSiblingIndex = lLastWasLeft ? NODE.right : NODE.left;
    SPINN_LOCK(lCurrSiblingIndex);

    lCurrSibling = &_bvh[lCurrSiblingIndex];

    lBBox.mergeWith(lCurrSibling->bbox);
    lSArea           = lBBox.surfaceArea();
    NODE.bbox        = lBBox;
    NODE.surfaceArea = lSArea;

    lSum             = lSum + SUM_OF(lCurrSiblingIndex) + lSArea * getCostInner();
    lMin             = min(lMin, MIN_OF(lCurrSiblingIndex));
    lNum             = lNum + lCurrSibling->numChildren + 2;
    SUM_OF(lNode)    = lSum;
    MIN_OF(lNode)    = lMin;
    NODE.numChildren = lNum;

    RELEASE_LOCK(lCurrSiblingIndex);

    if (lNode == _bvh.root()) { break; } // We processed the root ==> everything is done

    lLastWasLeft = NODE.isLeftChild();
    lLastIndex   = lNode;
    lLast        = &_bvh[lLastIndex];
    lNode        = NODE.parent;

    RELEASE_LOCK(lLastIndex);
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
    while (!NODE.isLeaf()) {
      lBitStack <<= 1;
      lBitStack |= 1;
      lNode = NODE.left;
    }

    // Leaf
    SUM_OF(lNode) = NODE.surfaceArea * getCostTri();
    MIN_OF(lNode) = NODE.surfaceArea;

    // Backtrack if left and right children are processed
    while ((lBitStack & 1) == 0) {
      if (lBitStack == 0 && lNode == lRoot) { return; } // We are done
      lNode = NODE.parent;

      AABB lBBox = LEFT.bbox;
      lBBox.mergeWith(RIGHT.bbox);
      float lSArea     = lBBox.surfaceArea();
      NODE.bbox        = lBBox;
      NODE.surfaceArea = lSArea;
      NODE.numChildren = LEFT.numChildren + RIGHT.numChildren + 2;
      SUM_OF(lNode)    = SUM_OF(NODE.left) + SUM_OF(NODE.right) + (lSArea * getCostInner());
      MIN_OF(lNode)    = min(MIN_OF(NODE.left), MIN_OF(NODE.right));
      lBitStack >>= 1;
    }

    lNode = PARENT.right;
    lBitStack ^= 1;
  }
}




ErrorCode Bittner13Par::runImpl(State &_state) {
  typedef tuple<uint32_t, float> TUP;

  unique_ptr<SumMin[]> lSumMin(new SumMin[_state.bvh.size()]);
  SumMin *             _sumMin = lSumMin.get();
  initSumAndMin(_state.bvh, _sumMin);

  uint32_t lNumNodes  = static_cast<uint32_t>((vBatchPercent / 100.0f) * static_cast<float>(_state.bvh.size()));
  uint32_t lChunkSize = lNumNodes / vNumChunks;
  auto     lComp      = [](TUP const &_l, TUP const &_r) -> bool { return get<1>(_l) > get<1>(_r); };

  vector<TUP>      lTodoList;
  vector<PATCH>    lPatches;
  vector<bool>     lSkipp;
  vector<uint32_t> lFixList;
  lTodoList.resize(_state.bvh.size());
  lPatches.resize(lChunkSize, PATCH(&_state.bvh));
  lSkipp.resize(lChunkSize);
  lFixList.resize(lChunkSize * 3);

  uint32_t lSkipped = 0;

  // Init List
#pragma omp parallel for
  for (uint32_t i = 0; i < lTodoList.size(); ++i) {
    lTodoList[i] = {i, 0.0f};
    ATO_OF(i).clear(memory_order_release);
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
        auto [lNodeIndex, _]                 = lTodoList[j * lChunkSize + k];
        auto [lRes, lTInsert, lTUnused, lGP] = removeNode(lNodeIndex, lPatches[k], _sumMin);
        auto [l1stIndex, l2ndIndex]          = lTInsert;
        auto [lU1, lU2]                      = lTUnused;

        if (!lRes) {
          lSkipp[k] = true;
          continue;
        }

        bool lR1 = reinsert(l1stIndex, lU1, lPatches[k], true, _sumMin);
        bool lR2 = reinsert(l2ndIndex, lU2, lPatches[k], false, _sumMin);
        if (!lR1 || !lR2) {
          lSkipp[k] = true;

          // Unlock Nodes
          ATO_OF(lNodeIndex).clear(memory_order_release);
          ATO_OF(l1stIndex).clear(memory_order_release);
          ATO_OF(l2ndIndex).clear(memory_order_release);
          ATO_OF(lU1).clear(memory_order_release);
          ATO_OF(lU2).clear(memory_order_release);
          ATO_OF(lGP).clear(memory_order_release);
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
        // Reset locks
        for (uint32_t l = 0; l < 10; ++l) {
          if (l >= lPatches[k].size()) { break; }
          ATO_OF(lPatches[k].getPatchedNodeIndex(l)).clear(memory_order_release);
        }

        if (lSkipp[k]) { continue; }
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
