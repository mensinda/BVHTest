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

  vMaxIntersections = _j.value("maxIntersections", vMaxIntersections);
}

json TracerBase::toJSON() const {
  return json{{"lightLocation", {{"x", vLightLocation.x}, {"y", vLightLocation.y}, {"z", vLightLocation.z}}},
              {"maxIntersections", vMaxIntersections}};
}

bool checkDir(fs::path _p) {
  auto lLogger = getLogger();
  if (!fs::exists(_p)) { fs::create_directory(_p); }
  if (!fs::is_directory(_p)) {
    lLogger->error("Path {} is not a directory", _p.string());
    return false;
  }
  return true;
}

// Color gradiant from https://stackoverflow.com/questions/22607043/color-gradient-algorithm
constexpr vec3 InverseSrgbCompanding(vec3 c) {
  // Inverse Red, Green, and Blue
  // clang-format off
  if (c.r > 0.04045) { c.r = pow((c.r + 0.055) / 1.055, 2.4); } else { c.r /= 12.92; }
  if (c.g > 0.04045) { c.g = pow((c.g + 0.055) / 1.055, 2.4); } else { c.g /= 12.92; }
  if (c.b > 0.04045) { c.b = pow((c.b + 0.055) / 1.055, 2.4); } else { c.b /= 12.92; }
  // clang-format on

  return c;
}

constexpr vec3 SrgbCompanding(vec3 c) {
  // Apply companding to Red, Green, and Blue
  // clang-format off
  if (c.r > 0.0031308) { c.r = 1.055 * pow(c.r, 1 / 2.4) - 0.055; } else { c.r *= 12.92; }
  if (c.g > 0.0031308) { c.g = 1.055 * pow(c.g, 1 / 2.4) - 0.055; } else { c.g *= 12.92; }
  if (c.b > 0.0031308) { c.b = 1.055 * pow(c.b, 1 / 2.4) - 0.055; } else { c.b *= 12.92; }
  // clang-format on

  return c;
}

const vec3 gBlue   = InverseSrgbCompanding({0.0f, 0.0f, 1.0f});
const vec3 gCyan   = InverseSrgbCompanding({0.0f, 1.0f, 1.0f});
const vec3 gGreen  = InverseSrgbCompanding({0.0f, 1.0f, 0.0f});
const vec3 gYellow = InverseSrgbCompanding({1.0f, 1.0f, 0.0f});
const vec3 gRed    = InverseSrgbCompanding({1.0f, 0.0f, 0.0f});

const uint32_t gNumColors = 5;
const vec3     gColors[5] = {gBlue, gCyan, gGreen, gYellow, gRed};

vec3 genColor(float _value) {
  uint32_t lInd1 = 0;
  uint32_t lInd2 = 0;
  float    lFrac = 0.0f;

  if (_value >= 1.0f) {
    lInd1 = lInd2 = gNumColors - 1;
  } else if (_value <= 0.0f) {
    lInd1 = lInd2 = 0;
  } else {
    _value *= gNumColors - 1;
    lInd1 = floor(_value);
    lInd2 = lInd1 + 1;
    lFrac = _value - static_cast<float>(lInd1);
  }

  vec3 lRes = gColors[lInd1] * (1 - lFrac) + gColors[lInd2] * lFrac;
  return SrgbCompanding(lRes);
}


ErrorCode TracerBase::writeImage(
    const vector<Pixel> &_pixels, uint32_t _width, uint32_t _height, string _name, State &_state) {

  auto     lLogger  = getLogger();
  fs::path lBaseDir = fs::absolute(_state.basePath) / _state.input;
  fs::path lOutDir  = lBaseDir / "images";

  if (!checkDir(lBaseDir)) { return ErrorCode::IO_ERROR; }
  if (!checkDir(lOutDir)) { return ErrorCode::IO_ERROR; }
  lOutDir = fs::canonical(fs::absolute(lOutDir));

  fs::path lColorPath = lOutDir / (_name + "_color.ppm");
  fs::path lInterPath = lOutDir / (_name + "_inter.ppm");

  fstream lColorFile(lColorPath.string(), lColorFile.out | lColorFile.binary);
  fstream lInterFile(lInterPath.string(), lInterFile.out | lInterFile.binary);

  if (!lColorFile.is_open()) {
    lLogger->error("Failed to open {} for writing", lColorPath.string());
    return ErrorCode::IO_ERROR;
  }

  if (!lInterFile.is_open()) {
    lLogger->error("Failed to open {} for writing", lInterPath.string());
    return ErrorCode::IO_ERROR;
  }

  lColorFile << "P6" << endl << _width << " " << _height << endl << 255 << endl;
  lInterFile << "P6" << endl << _width << " " << _height << endl << 255 << endl;

  for (auto const &i : _pixels) {
    lColorFile << i.r << i.g << i.b;
  }

  uint8_t lR;
  uint8_t lG;
  uint8_t lB;
  vec3    lTemp;
  for (auto const &i : _pixels) {
    lTemp = genColor(static_cast<float>(i.intCount) / static_cast<float>(vMaxIntersections));
    lR    = static_cast<uint8_t>(lTemp.r * 255);
    lG    = static_cast<uint8_t>(lTemp.g * 255);
    lB    = static_cast<uint8_t>(lTemp.b * 255);
    lInterFile << lR << lG << lB;
  }

  lColorFile.close();
  lInterFile.close();

  return ErrorCode::OK;
}
