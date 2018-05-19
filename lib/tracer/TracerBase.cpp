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
#include "TracerBase.hpp"

#include <fstream>

#if __has_include(<filesystem>)
#  include <filesystem>
namespace fs = std::filesystem;
#elif __has_include(<experimental/filesystem>)
#  include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#  error "std filesystem is not supported"
#endif

using namespace std;
using namespace BVHTest;
using namespace BVHTest::tracer;
using namespace BVHTest::base;

TracerBase::~TracerBase() {}

void TracerBase::fromJSON(const json &_j) {
  json tmp = _j; // Make a non const copy
  if (tmp.count("lightLocation") == 0) { tmp["lightLocation"] = json::object(); }

  vLightLocation.x = tmp["lightLocation"].value("x", vLightLocation.x);
  vLightLocation.y = tmp["lightLocation"].value("y", vLightLocation.y);
  vLightLocation.z = tmp["lightLocation"].value("z", vLightLocation.z);
}

json TracerBase::toJSON() const {
  return json{{"lightLocation", {{"x", vLightLocation.x}, {"y", vLightLocation.y}, {"z", vLightLocation.z}}}};
}
