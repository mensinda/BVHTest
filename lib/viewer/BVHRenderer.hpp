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

#include "base/BVH.hpp"
#include "RendererBase.hpp"

namespace BVHTest::view {

class BVHRenderer : public RendererBase {
 private:
  GLint vUniformLoc = 0;

  size_t vNumIndex = 0;

 public:
  BVHRenderer() = delete;
  BVHRenderer(base::BVH &_bvh);
  ~BVHRenderer();

  void     render() override;
  void     update(base::CameraBase *_cam) override;
  Renderer getType() const override { return Renderer::BVH; }
};

} // namespace BVHTest::view
