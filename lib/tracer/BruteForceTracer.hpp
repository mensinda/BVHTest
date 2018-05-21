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

class BruteForceTracer final : public TracerBase {
 private:
  base::Pixel trace(base::Ray &_ray, base::Mesh const &_mesh, base::BVH &_bvh);

 public:
  BruteForceTracer() = default;
  virtual ~BruteForceTracer();

  inline std::string getName() const override { return "bruteForce"; }
  inline std::string getDesc() const override { return "brute force raytracing. Bring lots of patiens"; }
  base::ErrorCode    runImpl(base::State &_state) override;
};

} // namespace BVHTest::tracer
