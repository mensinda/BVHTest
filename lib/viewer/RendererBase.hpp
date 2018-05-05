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

#include <glm/mat4x4.hpp>
#include "gl3w.h"

namespace BVHTest::view {

enum class Renderer { MESH, BVH };

class RendererBase {
 private:
  GLuint vVAO = 0;
  GLuint vVBO = 0;
  GLuint vEBO = 0;

  GLuint vVertexShader   = 0;
  GLuint vFragmentShader = 0;
  GLuint vShaderProg     = 0;

 protected:
  inline void bindVAO() { glBindVertexArray(vVAO); }
  inline void bindVBO() { glBindBuffer(GL_ARRAY_BUFFER, vVBO); }
  inline void bindEBO() { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vEBO); }
  inline void useProg() { glUseProgram(vShaderProg); }

  inline void unbindVAO() { glBindVertexArray(0); }

  bool  compileShaders(const char *_vert, const char *_frag);
  GLint getLocation(const char *_name);

 public:
  RendererBase();
  virtual ~RendererBase();

  virtual void     render()               = 0;
  virtual void     update(glm::mat4 _mvp) = 0;
  virtual Renderer getType() const        = 0;
};

} // namespace BVHTest::view
