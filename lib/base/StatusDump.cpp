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
#include "StatusDump.hpp"

using namespace BVHTest::base;
using namespace std;
using namespace std::chrono;

StatusDump::~StatusDump() {}

void StatusDump::fromJSON(const json &) {}
json StatusDump::toJSON() const { return json::object(); }

typedef milliseconds ms;

ErrorCode StatusDump::runImpl(State &_state) {
  auto lLogger = getLogger();

  lLogger->info("Status:");
  lLogger->info("  -- BVH:");
  lLogger->info("    - Nodes:       {}", _state.bvh.size());
  lLogger->info("    - Tree height: {}", _state.bvh.maxLevel());
  lLogger->info("    - SAH:         {}", _state.bvh.calcSAH());
  lLogger->info("  -- Mesh info:");
  lLogger->info("    - Vertices: {}", _state.mesh.vert.size());
  lLogger->info("    - Faces:    {}", _state.mesh.faces.size());
  lLogger->info("  -- Num Images: {}", _state.work.size());
  for (auto &i : _state.work) {
    milliseconds lMS = duration_cast<milliseconds>(i.time);
    nanoseconds  lNS = duration_cast<nanoseconds>(i.time - lMS);
    lLogger->info("    - Image {:<16} -- Time: {:>4}ms {:>8}ns", i.img.name, lMS.count(), lNS.count());
  }
  lLogger->info("  -- Command times:");
  for (auto const &i : _state.commands) {
    seconds lSec = duration_cast<seconds>(i.duration);
    lLogger->info("    - {:<16} -- {:>3}s {:>3}ms", i.name, lSec.count(), duration_cast<ms>(i.duration - lSec).count());
  }

  lLogger->info("");
  return ErrorCode::OK;
}
