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

#include "TracerBase.hpp"

namespace BVHTest::tracer {

class CUDATracer final : public TracerBase {
  base::Ray *            vRays       = nullptr;
  base::CameraBase::RES  vResolution = {0, 0};
  std::vector<uint8_t *> vDeviceImages;

 public:
  CUDATracer() = default;
  virtual ~CUDATracer();

  bool allocateMemory(base::CameraBase::RES _res, uint32_t _numImages);
  void freeMemory();

  inline std::string getName() const override { return "CUDATracer"; }
  inline std::string getDesc() const override { return "stackless BVH CUDA Ray tracer"; }
  base::ErrorCode    setup(base::State &_state) override;
  base::ErrorCode    runImpl(base::State &_state) override;
  void               teardown(base::State &_state) override;

  inline uint64_t getRequiredCommands() const override {
    return TracerBase::getRequiredCommands() | static_cast<uint64_t>(base::CommandType::CUDA_INIT);
  }
};

} // namespace BVHTest::tracer
