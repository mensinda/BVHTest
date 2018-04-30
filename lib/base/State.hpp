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

#pragma once

#include <stdint.h>

namespace BVHTest::base {

enum class CommandType {
  IMPORT    = (1 << 0), // Data was imported
  BVH_BUILD = (1 << 4),
  BVH_OPT1  = (1 << 5),
  BVH_OPT2  = (1 << 6),
  RAY_TRACE = (1 << 16),
  EXPORT    = (1 << 24) // Data was exported
};

struct State final {
  uint64_t commandsRun = 0;
};

} // namespace BVHTest::base
