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

class MeshRenderer : public RendererBase {
 private:
  GLint vUniformLoc = 0;

  size_t vNumIndex = 0;
  size_t vNumVert  = 0;

  base::MeshRaw vRawMesh;

  glm::vec3 *vDevOriginalVert = nullptr;

 public:
  MeshRenderer() = delete;
  MeshRenderer(base::State &_state);
  ~MeshRenderer();

  void        render() override;
  void        update(base::CameraBase *_cam) override;
  void        updateMesh(base::State &_state, base::CameraBase *_cam, uint32_t _offsetIndex) override;
  Renderer    getType() const override { return Renderer::MESH; }
  uint32_t    numRenderModes() override { return 2; }
  std::string getRenderModeString() override;
};

} // namespace BVHTest::view
