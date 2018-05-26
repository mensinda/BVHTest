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

#include "BVHRenderer.hpp"
#include "misc/Camera.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>

using namespace glm;
using namespace std;
using namespace BVHTest;
using namespace BVHTest::view;
using namespace BVHTest::base;
using namespace BVHTest::misc;

struct VBOData {
  vec3 pos;
  vec3 color;
};

struct Line {
  uint32_t start;
  uint32_t end;
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

#define V0 lVOffset + 0
#define V1 lVOffset + 1
#define V2 lVOffset + 2
#define V3 lVOffset + 3
#define V4 lVOffset + 4
#define V5 lVOffset + 5
#define V6 lVOffset + 6
#define V7 lVOffset + 7

inline void addAABB(AABB const &_aabb, size_t _num, vec3 _color, vector<VBOData> &_vert, vector<Line> &_ind) {
  uint32_t lVOffset = _num * 8;
  uint32_t lIOffset = _num * 12;

  vec3 const &min = _aabb.min;
  vec3 const &max = _aabb.max;

  _vert[V0]           = {{min.x, min.y, min.z}, _color};
  _vert[V1]           = {{max.x, min.y, min.z}, _color};
  _vert[V2]           = {{min.x, max.y, min.z}, _color};
  _vert[V3]           = {{min.x, min.y, max.z}, _color};
  _vert[V4]           = {{max.x, max.y, min.z}, _color};
  _vert[V5]           = {{min.x, max.y, max.z}, _color};
  _vert[V6]           = {{max.x, min.y, max.z}, _color};
  _vert[V7]           = {{max.x, max.y, max.z}, _color};
  _ind[lIOffset + 0]  = {V0, V1};
  _ind[lIOffset + 1]  = {V0, V2};
  _ind[lIOffset + 2]  = {V0, V3};
  _ind[lIOffset + 3]  = {V1, V4};
  _ind[lIOffset + 4]  = {V1, V6};
  _ind[lIOffset + 5]  = {V2, V4};
  _ind[lIOffset + 6]  = {V2, V5};
  _ind[lIOffset + 7]  = {V3, V5};
  _ind[lIOffset + 8]  = {V3, V6};
  _ind[lIOffset + 9]  = {V4, V7};
  _ind[lIOffset + 10] = {V5, V7};
  _ind[lIOffset + 11] = {V6, V7};
}

// Color gradiant from https://stackoverflow.com/questions/22607043/color-gradient-algorithm
constexpr vec3 InverseSrgbCompanding(vec3 c) {
  // Inverse Red, Green, and Blue
  // clang-format off
  if (c.r > 0.04045) { c.r = pow((c.r + 0.055) / 1.055, 2.4); } else { c.r /= 12.92; }
  if (c.g > 0.04045) { c.g = pow((c.g + 0.055) / 1.055, 2.4); } else { c.g /= 12.92; }
  if (c.b > 0.04045) { c.b = pow((c.b + 0.055) / 1.055, 2.4); } else { c.b /= 12.92; }
  // clang-format on

  return c;
}

constexpr vec3 SrgbCompanding(vec3 c) {
  // Apply companding to Red, Green, and Blue
  // clang-format off
  if (c.r > 0.0031308) { c.r = 1.055 * pow(c.r, 1 / 2.4) - 0.055; } else { c.r *= 12.92; }
  if (c.g > 0.0031308) { c.g = 1.055 * pow(c.g, 1 / 2.4) - 0.055; } else { c.g *= 12.92; }
  if (c.b > 0.0031308) { c.b = 1.055 * pow(c.b, 1 / 2.4) - 0.055; } else { c.b *= 12.92; }
  // clang-format on

  return c;
}

const vec3 gBlue   = InverseSrgbCompanding({0.0f, 0.0f, 1.0f});
const vec3 gCyan   = InverseSrgbCompanding({0.0f, 1.0f, 1.0f});
const vec3 gGreen  = InverseSrgbCompanding({0.0f, 1.0f, 0.0f});
const vec3 gYellow = InverseSrgbCompanding({1.0f, 1.0f, 0.0f});
const vec3 gRed    = InverseSrgbCompanding({1.0f, 0.0f, 0.0f});

const uint32_t gNumColors = 5;
const vec3     gColors[5] = {gBlue, gCyan, gGreen, gYellow, gRed};

vec3 genColor(uint32_t _level, uint32_t _maxLevel) {
  float    lValue = static_cast<float>(_level) / static_cast<float>(_maxLevel);
  uint32_t lInd1  = 0;
  uint32_t lInd2  = 0;
  float    lFrac  = 0.0f;

  if (lValue >= 1.0f) {
    lInd1 = lInd2 = gNumColors - 1;
  } else if (lValue <= 0.0f) {
    lInd1 = lInd2 = 0;
  } else {
    lValue *= gNumColors - 1;
    lInd1 = floor(lValue);
    lInd2 = lInd1 + 1;
    lFrac = lValue - static_cast<float>(lInd1);
  }

  vec3 lRes = gColors[lInd1] * (1 - lFrac) + gColors[lInd2] * lFrac;
  return SrgbCompanding(lRes);
}

uint32_t processNode(BVH &_bvh, uint32_t _node, uint32_t _next, vector<VBOData> &_vert, vector<Line> &_ind) {
  BVHNode const &lNode = _bvh[_node];

  addAABB(lNode.bbox, _next++, genColor(lNode.level, _bvh.maxLevel()), _vert, _ind);

  if (lNode.isLeaf()) { return _next; }

  if (lNode.left != UINT32_MAX) { _next = processNode(_bvh, lNode.left, _next, _vert, _ind); }
  if (lNode.right != UINT32_MAX) { _next = processNode(_bvh, lNode.right, _next, _vert, _ind); }

  return _next;
}

BVHRenderer::BVHRenderer(BVH &_bvh) {
  // Generate OpenGL VBO data
  std::vector<VBOData> lVert;
  std::vector<Line>    lIndex;
  lVert.resize(_bvh.size() * 8);
  lIndex.resize(_bvh.size() * 12);

  uint32_t lNumGenerated = processNode(_bvh, _bvh.root(), 0, lVert, lIndex);

  lVert.resize(lNumGenerated * 8);
  lIndex.resize(lNumGenerated * 12);

  bindVAO();
  bindVBO();
  glBufferData(GL_ARRAY_BUFFER, lVert.size() * sizeof(VBOData), lVert.data(), GL_STATIC_DRAW);

  bindEBO();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, lIndex.size() * sizeof(Line), lIndex.data(), GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VBOData), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VBOData), (void *)offsetof(VBOData, color));

  unbindVAO();

  if (!compileShaders(gVertexShader, gFragmentShader)) { return; }

  vNumIndex   = lIndex.size() * 2;
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
