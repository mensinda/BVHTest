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
#include "BVHRenderer.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::view;
using namespace BVHTest::base;

using namespace std;
using namespace BVHTest;
using namespace BVHTest::view;
using namespace BVHTest::base;

struct VBOData {
  glm::vec3 vert;
  glm::vec3 norm;
};

static const char *gVertexShader = R"__GLSL__(
#version 330 core

layout (location = 0) in vec3 iVert;
layout (location = 1) in vec3 iNorm;

uniform mat4 uMVP;

out vec4 vNorm;

void main() {
  vNorm = vec4(iNorm.xyz, 1.0);
  gl_Position = uMVP * vec4(iVert.xyz, 1.0);
}
)__GLSL__";

static const char *gFragmentShader = R"__GLSL__(
#version 330 core

in  vec4 vNorm;
out vec4 oColor;

void main() {
  oColor = vec4(normalize(vNorm).xyz, 1.0);
}
)__GLSL__";


BVHRenderer::BVHRenderer(const Mesh &_mesh) {
  // Generate OpenGL VBO data
  std::vector<VBOData> lOGLData;
  lOGLData.resize(_mesh.vert.size());
  for (size_t i = 0; i < _mesh.vert.size(); ++i) {
    lOGLData[i].vert = _mesh.vert[i];
    lOGLData[i].norm = _mesh.norm[i];
  }

  bindVAO();
  bindVBO();
  glBufferData(GL_ARRAY_BUFFER, lOGLData.size() * sizeof(VBOData), lOGLData.data(), GL_STATIC_DRAW);

  bindEBO();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, _mesh.faces.size() * sizeof(Triangle), _mesh.faces.data(), GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VBOData), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VBOData), (void *)offsetof(VBOData, norm));

  unbindVAO();

  if (!compileShaders(gVertexShader, gFragmentShader)) { return; }

  vNumIndex   = _mesh.faces.size() * 3;
  vUniformLoc = getLocation("uMVP");
}

BVHRenderer::~BVHRenderer() {}

void BVHRenderer::update(glm::mat4 _mvp) {
  useProg();
  glUniformMatrix4fv(vUniformLoc, 1, GL_FALSE, glm::value_ptr(_mvp));
}

void BVHRenderer::render() {
  useProg();
  bindVAO();
  glDrawElements(GL_TRIANGLES, vNumIndex, GL_UNSIGNED_INT, 0);
  unbindVAO();
}
