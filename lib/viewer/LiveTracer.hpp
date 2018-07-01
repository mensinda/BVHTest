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
#include "RendererBase.hpp"

namespace BVHTest::view {

class LiveTracer : public RendererBase {
 private:
  GLuint                  vTexture;
  void *                  vCudaRes     = nullptr;
  uint8_t *               vDeviceImage = nullptr;
  base::Ray *             vRays        = nullptr;
  base::CameraBase *      vCam         = nullptr;
  base::State::CudaMemory vCudaMem;

  uint32_t vWidth;
  uint32_t vHeight;

  static bool vCudaIsInit;

 public:
  LiveTracer(base::State &_state, uint32_t _w, uint32_t _h);
  virtual ~LiveTracer();

  static bool cudaInit();

  void     render() override;
  void     update(base::CameraBase *_cam) override;
  Renderer getType() const override { return Renderer::CUDA_TRACER; }
};

} // namespace BVHTest::view