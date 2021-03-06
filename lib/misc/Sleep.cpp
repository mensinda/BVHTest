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

#include "Sleep.hpp"
#include <thread>

using namespace std;
using namespace std::chrono;
using namespace BVHTest::base;
using namespace BVHTest::misc;

Sleep::~Sleep() {}

void Sleep::fromJSON(const json &_j) { vSleepTime = milliseconds(_j.value("durationMS", vSleepTime.count())); }
json Sleep::toJSON() const { return {{"durationMS", (uint64_t)vSleepTime.count()}}; }

ErrorCode Sleep::runImpl(base::State &) {
  this_thread::sleep_for(vSleepTime);
  return ErrorCode::OK;
}
