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

#include "base/State.hpp"
#include "tracer/CUDAKernels.hpp"
#include "RendererBase.hpp"
#include <chrono>

namespace BVHTest::view {

class LiveTracer : public RendererBase {
 public:
  typedef std::chrono::time_point<std::chrono::system_clock> TimePoint;

 private:
  GLuint                  vTexture;
  void *                  vCudaRes     = nullptr;
  CUDAPixel *             vDeviceImage = nullptr;
  base::Ray *             vRays        = nullptr;
  base::CameraBase *      vCam         = nullptr;
  base::State::CudaMemory vCudaMem;

  uint32_t vWidth;
  uint32_t vHeight;

  bool vBundle  = false;
  bool vBVHView = false;

  GLint vIntCountLocation = 0;
  GLint vMaxCountLocation = 0;

  TimePoint vPercentileRecalc = std::chrono::system_clock::now();
  uint32_t  vMaxCount         = 255;

 public:
  LiveTracer(base::State &_state, uint32_t _w, uint32_t _h);
  virtual ~LiveTracer();

  static bool cudaInit();

  void        render() override;
  void        update(base::CameraBase *_cam) override;
  Renderer    getType() const override { return Renderer::CUDA_TRACER; }
  uint32_t    numRenderModes() override { return 4; }
  std::string getRenderModeString() override;
};

} // namespace BVHTest::view
