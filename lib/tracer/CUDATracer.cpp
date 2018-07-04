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

#include "CUDATracer.hpp"
#include "misc/Camera.hpp"
#include "CUDAKernels.hpp"
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace BVHTest;
using namespace BVHTest::tracer;
using namespace BVHTest::base;
using namespace BVHTest::misc;

CUDATracer::~CUDATracer() { freeMemory(); }

void CUDATracer::fromJSON(const json &_j) {
  TracerBase::fromJSON(_j);
  vRayBundles = _j.value("rayBundles", vRayBundles);
}

json CUDATracer::toJSON() const {
  json js          = TracerBase::toJSON();
  js["rayBundles"] = vRayBundles;
  return js;
}


bool CUDATracer::allocateMemory(CameraBase::RES _res, uint32_t _numImages) {
  freeMemory();
  vResolution = _res;
  bool lRes   = allocateRays(&vRays, _res.width * _res.height);

  if (!lRes) {
    vResolution = {0, 0};
    return false;
  }

  for (uint32_t i = 0; i < _numImages; ++i) {
    uint8_t *lCUDAImage = nullptr;
    lRes                = allocateImage(&lCUDAImage, vResolution.width, vResolution.height);
    if (!lRes) {
      freeMemory();
      return false;
    }
    vDeviceImages.push_back(lCUDAImage);
  }

  return lRes;
}

void CUDATracer::freeMemory() {
  freeRays(&vRays);
  vResolution = {0, 0};

  for (uint8_t *i : vDeviceImages) { freeImage(i); }
  vDeviceImages.clear();
}


ErrorCode CUDATracer::setup(State &_state) {
  uint32_t lNumImages = _state.cameras.size();
  if (lNumImages == 0) { return ErrorCode::WARNING; }

  auto lRes = _state.cameras[0]->getResolution();
  if (!allocateMemory(lRes, lNumImages)) { return ErrorCode::CUDA_ERROR; }

  return ErrorCode::OK;
}


ErrorCode CUDATracer::runImpl(State &_state) {
  uint32_t lOffset = _state.work.size();
  auto     lRes    = _state.cameras[0]->getResolution();
  _state.work.resize(lOffset + _state.cameras.size());

  for (uint32_t i = lOffset; i < _state.work.size(); ++i) {
    auto &  lCurr = _state.work[i];
    Camera *lCam  = dynamic_cast<Camera *>(_state.cameras[i - lOffset].get());
    if (!lCam) { continue; }

    PROGRESS("Tracing Image: " + to_string(i - lOffset), i - lOffset, _state.work.size() - lOffset - 1);

    auto lCamData = lCam->getCamera();
    auto lStart   = high_resolution_clock::now();

    generateRays(vRays, lRes.width, lRes.height, lCamData.pos, lCamData.lookAt, lCamData.up, lCamData.fov);
    tracerImage(vRays,
                vDeviceImages[i - lOffset],
                _state.cudaMem.bvh.nodes,
                _state.bvh.root(),
                _state.cudaMem.rawMesh,
                getLightLocation(),
                lRes.width,
                lRes.height,
                vRayBundles);
    tracerDoCudaSync();

    auto lEnd  = high_resolution_clock::now();
    lCurr.time = duration_cast<nanoseconds>(lEnd - lStart);
  }

  return ErrorCode::OK;
}

void CUDATracer::teardown(State &_state) {
  uint32_t               lNumPixel = vResolution.width * vResolution.height;
  std::vector<CUDAPixel> lPixels;
  lPixels.resize(lNumPixel);

  for (uint32_t i = 0; i < vDeviceImages.size(); ++i) {
    Image &lIMG = _state.work[_state.work.size() - vDeviceImages.size() + i].img;
    copyImageToHost(lPixels.data(), vDeviceImages[i], vResolution.width, vResolution.height);

    lIMG.name   = _state.name + "_CUDA_cam_" + to_string(i);
    lIMG.width  = vResolution.width;
    lIMG.height = vResolution.height;
    lIMG.pixels.resize(lNumPixel);
#pragma omp parallel for
    for (uint32_t j = 0; j < lNumPixel; ++j) {
      if (lPixels[j].hit != 0) {
        lIMG.pixels[j].r = lPixels[j].diffuse;
        lIMG.pixels[j].g = lPixels[j].diffuse;
        lIMG.pixels[j].b = lPixels[j].diffuse;
      } else {
        lIMG.pixels[j].r = 121;
        lIMG.pixels[j].g = 167;
        lIMG.pixels[j].b = 229;
      }
      lIMG.pixels[j].intCount = lPixels[j].intCount;
      lIMG.pixels[j].rayTime  = 0;
    }
  }

  freeMemory();
}
