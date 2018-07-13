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

#include "OptimizerBase.hpp"

using namespace std;
using namespace std::chrono;
using namespace std::chrono_literals;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::builder;
using namespace BVHTest::base;

OptimizerBase::~OptimizerBase() {}
void OptimizerBase::fromJSON(const json &_j) {
  vCalcSAH = _j.value("base_calcSAH", vCalcSAH);
  vDoSync  = _j.value("base_doSync", vDoSync);
}

json OptimizerBase::toJSON() const { return json{{"base_calcSAH", vCalcSAH}, {"base_doSync", vDoSync}}; }

void OptimizerBase::benchmarkInitData(State &_state, function<float()> _calcSAH, function<void()> _sync) {
  _state.optData.numSkipped = 0;
  _state.optData.numTotal   = 0;
  _state.optData.optStepps.push_back({0, 0.0f, 0ms});
  if (vCalcSAH) { _state.optData.optStepps.back().sah = _calcSAH(); }
  if (vDoSync) { _sync(); }
}

void OptimizerBase::benchmarkStartTimer(State &_state, function<void()> _sync) {
  _state.optData.optStepps.push_back({(uint32_t)_state.optData.optStepps.size(), 0.0f, 0ms});
  if (vDoSync) { _sync(); }
  vStart = chrono::system_clock::now();
}


void OptimizerBase::benchmarkRecordData(State &_state, function<float()> _calcSAH, function<void()> _sync) {
  if (vDoSync) { _sync(); }
  auto lEnd = chrono::system_clock::now();

  _state.optData.optStepps.back().duration = duration_cast<milliseconds>(lEnd - vStart);

  if (vCalcSAH) { _state.optData.optStepps.back().sah = _calcSAH(); }
}
