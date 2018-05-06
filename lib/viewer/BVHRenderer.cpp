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
#include "camera/Camera.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>

using namespace glm;
using namespace std;
using namespace BVHTest;
using namespace BVHTest::view;
using namespace BVHTest::base;
using namespace BVHTest::camera;

struct VBOData {
  vec3 pos;
  vec3 color;
};

static const char *gVertexShader = R"__GLSL__(
#version 330 core

layout (location = 0) in vec3 iVert;
layout (location = 1) in vec3 iColor;

uniform mat4 uMVP;

out vec3 vColor;

void main() {
  vColor = iColor;
  gl_Position = uMVP * vec4(iVert.xyz, 1.0);
}
)__GLSL__";

static const char *gFragmentShader = R"__GLSL__(
#version 330 core

in  vec3 vColor;
out vec4 oColor;

void main() {
  oColor = vec4(vColor.xyz, 1.0);
}
)__GLSL__";

void addAABB(AABB const &_aabb, size_t _num, vec3 &_color, vector<VBOData> &_vert, vector<uint32_t> &_ind) {
  size_t lVOffset = _num * 8;
  size_t lIOffset = _num * 12 * 2;

  vec3 const &min = _aabb.min;
  vec3 const &max = _aabb.max;

  _vert[lVOffset + 0] = {{min.x, min.y, min.z}, _color};
  _vert[lVOffset + 0] = {{max.x, max.y, max.z}, _color};
}



BVHRenderer::BVHRenderer(std::vector<AABB> const &_bboxes) {
  // Generate OpenGL VBO data
  std::vector<VBOData>  lVert;
  std::vector<uint32_t> lIndex;
  lVert.resize(_bboxes.size() * 8);
  lIndex.resize(_bboxes.size() * 12 * 2);
  vec3 lColor = {1, 0, 0};
  for (size_t i = 0; i < _bboxes.size(); ++i) {
    addAABB(_bboxes[i], i, lColor, lVert, lIndex);
  }

  bindVAO();
  bindVBO();
  glBufferData(GL_ARRAY_BUFFER, lVert.size() * sizeof(VBOData), lVert.data(), GL_STATIC_DRAW);

  bindEBO();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, lIndex.size() * sizeof(uint32_t), lIndex.data(), GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VBOData), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VBOData), (void *)offsetof(VBOData, color));

  unbindVAO();

  if (!compileShaders(gVertexShader, gFragmentShader)) { return; }

  vNumIndex   = lIndex.size();
  vUniformLoc = getLocation("uMVP");
}

BVHRenderer::~BVHRenderer() {}

void BVHRenderer::update(CameraBase *_cam) {
  Camera *lCam = dynamic_cast<Camera *>(_cam);
  if (!lCam) { return; }

  useProg();
  glUniformMatrix4fv(vUniformLoc, 1, GL_FALSE, glm::value_ptr(lCam->getViewProjection()));
}

void BVHRenderer::render() {
  useProg();
  bindVAO();
  glDrawElements(GL_LINES, vNumIndex, GL_UNSIGNED_INT, 0);
  unbindVAO();
}
