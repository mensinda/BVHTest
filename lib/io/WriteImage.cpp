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
#include "WriteImage.hpp"
#include "lodepng/lodepng.h"
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
using namespace BVHTest::IO;
using namespace BVHTest::base;

WriteImage::~WriteImage() {}
void WriteImage::fromJSON(const json &_j) {
  vPercentile = _j.value("percentile", vPercentile);
  if (vPercentile <= 0.1f) { vPercentile = 0.1f; }
  if (vPercentile >= 99.9999f) { vPercentile = 99.9999f; }
}
json WriteImage::toJSON() const { return json{{"percentile", vPercentile}}; }

inline bool checkDir(fs::path _p) {
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

inline vec3 genColor(float _value) {
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


ErrorCode WriteImage::writePNG(string _name, vector<uint8_t> const &_data, uint32_t _w, uint32_t _h) {
  auto    lLogger = getLogger();
  fstream lFile(_name, lFile.out | lFile.binary);

  if (!lFile.is_open()) {
    lLogger->error("Failed to open '{}' for writing", _name);
    return ErrorCode::IO_ERROR;
  }

  vector<uint8_t> lOut;
  if (lodepng::encode(lOut, _data, _w, _h, LCT_RGB) != 0) {
    lLogger->error("PNG encoder error for '{}'", _name);
    return ErrorCode::GENERIC_ERROR;
  }

  lFile.write(reinterpret_cast<char *>(lOut.data()), lOut.size());
  lFile.close();

  return ErrorCode::OK;
}


ErrorCode WriteImage::runImpl(State &_state) {
  if (_state.work.empty()) { return ErrorCode::OK; }

  auto     lLogger  = getLogger();
  fs::path lBaseDir = fs::absolute(_state.basePath) / _state.input;
  fs::path lOutDir  = lBaseDir / "images";

  if (!checkDir(lBaseDir)) { return ErrorCode::IO_ERROR; }
  if (!checkDir(lOutDir)) { return ErrorCode::IO_ERROR; }
  lOutDir = fs::canonical(fs::absolute(lOutDir));

  size_t           lTotalSize = 0;
  size_t           lOffset    = 0;
  vector<uint32_t> lAllInter;
  vector<uint64_t> lAllRTime;

  for (auto &i : _state.work) {
    lTotalSize += i.img.pixels.size();
  }

  lAllInter.resize(lTotalSize);
  lAllRTime.resize(lTotalSize);

  for (auto &i : _state.work) {
    for (size_t j = 0; j < i.img.pixels.size(); ++j) {
      lAllInter[lOffset + j] = i.img.pixels[j].intCount;
      lAllRTime[lOffset + j] = i.img.pixels[j].rayTime;
    }
    lOffset += i.img.pixels.size();
  }

  uint32_t lOff = (end(lAllInter) - begin(lAllInter)) * (vPercentile / 100.0f);
  nth_element(begin(lAllInter), begin(lAllInter) + lOff, end(lAllInter));
  nth_element(begin(lAllRTime), begin(lAllRTime) + lOff, end(lAllRTime));
  uint32_t lMaxInter = lAllInter[lOff];
  uint32_t lMaxRTime = lAllRTime[lOff];

  uint32_t lCounter = 0;

  for (auto &i : _state.work) {
    PROGRESS("Writing Image " + to_string(lCounter), lCounter, _state.work.size() - 1);
    lCounter++;

    if (i.img.pixels.empty() || i.img.width == 0 || i.img.height == 0) { continue; }

    fs::path lColorPath = lOutDir / (i.img.name + "_color.png");
    fs::path lInterPath = lOutDir / (i.img.name + "_inter.png");
    fs::path lRTimePath = lOutDir / (i.img.name + "_rtime.png");

    vector<uint8_t> lColorData;
    vector<uint8_t> lInterData;
    vector<uint8_t> lRTimeData;

    auto const &lPX = i.img.pixels;

    lColorData.resize(lPX.size() * 3);
    lInterData.resize(lPX.size() * 3);
    lRTimeData.resize(lPX.size() * 3);

#pragma omp parallel for
    for (size_t j = 0; j < lPX.size(); ++j) {
      lColorData[j * 3 + 0] = lPX[j].r;
      lColorData[j * 3 + 1] = lPX[j].g;
      lColorData[j * 3 + 2] = lPX[j].b;

      vec3 lTemp            = genColor(static_cast<float>(lPX[j].intCount) / static_cast<float>(lMaxInter));
      lInterData[j * 3 + 0] = static_cast<uint8_t>(lTemp.r * 255);
      lInterData[j * 3 + 1] = static_cast<uint8_t>(lTemp.g * 255);
      lInterData[j * 3 + 2] = static_cast<uint8_t>(lTemp.b * 255);

      lTemp                 = genColor(static_cast<float>(lPX[j].rayTime) / static_cast<float>(lMaxRTime));
      lRTimeData[j * 3 + 0] = static_cast<uint8_t>(lTemp.r * 255);
      lRTimeData[j * 3 + 1] = static_cast<uint8_t>(lTemp.g * 255);
      lRTimeData[j * 3 + 2] = static_cast<uint8_t>(lTemp.b * 255);
    }

    auto lRes1 = writePNG(lColorPath.string(), lColorData, i.img.width, i.img.height);
    auto lRes2 = writePNG(lInterPath.string(), lInterData, i.img.width, i.img.height);
    auto lRes3 = writePNG(lRTimePath.string(), lRTimeData, i.img.width, i.img.height);

    if (lRes1 != ErrorCode::OK || lRes2 != ErrorCode::OK || lRes3 != ErrorCode::OK) { return ErrorCode::IO_ERROR; }
  }

  return ErrorCode::OK;
}
