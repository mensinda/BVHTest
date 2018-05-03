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
#include <glm/glm.hpp>
#include "gl3w.h"

namespace BVHTest::view {

class MeshRenderer {
 private:
  GLuint vVAO = 0;
  GLuint vVBO = 0;
  GLuint vEBO = 0;

  GLuint vVertexShader   = 0;
  GLuint vFragmentShader = 0;
  GLuint vShaderProg     = 0;

  GLint vUniformLoc = 0;

  size_t vNumIndex = 0;

 public:
  MeshRenderer() = delete;
  MeshRenderer(base::Mesh const &_mesh);
  ~MeshRenderer();

  void render();
  void update(glm::mat4 _mvp);
};

} // namespace BVHTest::view
