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

#include "LiveTracer.hpp"
#include "misc/Camera.hpp"
#include "tracer/CUDAKernels.hpp"
#include "tracer/CUDATracer.hpp"

using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::view;

static const char *gVertexShader = R"__GLSL__(
#version 330 core

layout (location = 0) in vec2 iVert;

out vec2 vTexCoord;

void main() {
  vTexCoord = vec2((iVert.x + 1) / 2, 1 - (iVert.y + 1) / 2);
  gl_Position = vec4(iVert.xy, 0.0, 1.0);
}
)__GLSL__";

static const char *gFragmentShader = R"__GLSL__(
#version 330 core

in  vec2 vTexCoord;
out vec4 oColor;

uniform usampler2D uTex;
uniform uint       uIntCount;
uniform uint       uMaxCount;

// Color gradiant from https://stackoverflow.com/questions/22607043/color-gradient-algorithm
vec3 InverseSrgbCompanding(vec3 c) {
  // Inverse Red, Green, and Blue
  // clang-format off
  if (c.r > 0.04045) { c.r = pow((c.r + 0.055) / 1.055, 2.4); } else { c.r /= 12.92; }
  if (c.g > 0.04045) { c.g = pow((c.g + 0.055) / 1.055, 2.4); } else { c.g /= 12.92; }
  if (c.b > 0.04045) { c.b = pow((c.b + 0.055) / 1.055, 2.4); } else { c.b /= 12.92; }
  // clang-format on

  return c;
}

vec3 SrgbCompanding(vec3 c) {
  // Apply companding to Red, Green, and Blue
  // clang-format off
  if (c.r > 0.0031308) { c.r = 1.055 * pow(c.r, 1 / 2.4) - 0.055; } else { c.r *= 12.92; }
  if (c.g > 0.0031308) { c.g = 1.055 * pow(c.g, 1 / 2.4) - 0.055; } else { c.g *= 12.92; }
  if (c.b > 0.0031308) { c.b = 1.055 * pow(c.b, 1 / 2.4) - 0.055; } else { c.b *= 12.92; }
  // clang-format on

  return c;
}

const vec3 gBlue   = vec3(0.0f, 0.0f, 1.0f);
const vec3 gCyan   = vec3(0.0f, 1.0f, 1.0f);
const vec3 gGreen  = vec3(0.0f, 1.0f, 0.0f);
const vec3 gYellow = vec3(1.0f, 1.0f, 0.0f);
const vec3 gRed    = vec3(1.0f, 0.0f, 0.0f);

const uint gNumColors = 5u;
const vec3 gColors[gNumColors] = vec3[](gBlue, gCyan, gGreen, gYellow, gRed);

vec3 genColor(float value) {
  uint  lInd1 = 0u;
  uint  lInd2 = 0u;
  float lFrac = 0.0f;

  if (value >= 1.0f) {
    lInd1 = lInd2 = gNumColors - 1u;
  } else if (value <= 0.0f) {
    lInd1 = lInd2 = 0u;
  } else {
    value *= gNumColors - 1u;
    lInd1 = uint(floor(value));
    lInd2 = lInd1 + 1u;
    lFrac = value - float(lInd1);
  }

  vec3 lRes = gColors[lInd1] * (1u - lFrac) + gColors[lInd2] * lFrac;
  return SrgbCompanding(lRes);
}

void main() {
  uvec4 lPixelData = texture(uTex, vTexCoord);
  if(uIntCount == 0u) {
    if(lPixelData.r == 0u) {
      oColor = vec4(0.4765625, 0.65625, 0.8984375, 1.0f);
    } else {
      oColor = vec4(lPixelData.g / 255.0f, lPixelData.g / 255.0f, lPixelData.g / 255.0f, 1.0f);
    }
  } else {
    oColor = vec4(genColor(((lPixelData.a << 8) + lPixelData.b) / float(uMaxCount)), 1.0f);
  }
}
)__GLSL__";

LiveTracer::LiveTracer(State &_state, uint32_t _w, uint32_t _h) {
  vWidth  = _w;
  vHeight = _h;

  allocateRays(&vRays, vWidth * vHeight);
  allocateImage(&vDeviceImage, vWidth, vHeight);

  float lVert[] = {1, 1, 1, -1, -1, 1, -1, -1};

  uint32_t lInd[] = {/* Tri 1 */ 2, 0, 1, /* Tri 2 */ 2, 1, 3};

  bindVAO();
  bindVBO();
  glBufferData(GL_ARRAY_BUFFER, 2 * 4 * sizeof(float), lVert, GL_STATIC_DRAW);

  bindEBO();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, 6 * sizeof(uint32_t), lInd, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), (void *)0);

  unbindVAO();

  glGenTextures(1, &vTexture);
  glBindTexture(GL_TEXTURE_2D, vTexture);

  // set basic parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

  // Create texture data (4-component unsigned byte)
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8UI, vWidth, vHeight, 0, GL_RGBA_INTEGER, GL_UNSIGNED_BYTE, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

  if (!compileShaders(gVertexShader, gFragmentShader)) { return; }

  vIntCountLocation = getLocation("uIntCount");
  vMaxCountLocation = getLocation("uMaxCount");

  registerOGLImage(&vCudaRes, vTexture);
  vCudaMem = _state.cudaMem;
}

LiveTracer::~LiveTracer() {
  unregisterOGLImage(vCudaRes);
  freeRays(&vRays);
  freeImage(vDeviceImage);
  glDeleteTextures(1, &vTexture);
}

void LiveTracer::render() {
  misc::Camera *lCam = dynamic_cast<misc::Camera *>(vCam);
  if (!vCam || !lCam) { return; }

  auto lCamData = lCam->getCamera();

  generateRays(vRays, vWidth, vHeight, lCamData.pos, lCamData.lookAt, lCamData.up, lCamData.fov);
  tracerImage(vRays, vDeviceImage, vCudaMem.bvh.nodes, 0, vCudaMem.rawMesh, vec3(1, 1, 1), vWidth, vHeight, vBundle);
  copyToOGLImage(&vCudaRes, vDeviceImage, vWidth, vHeight);

  useProg();
  glUniform1ui(vIntCountLocation, vBVHView ? 1u : 0u);
  glUniform1ui(vMaxCountLocation, vMaxCount);
  glBindTexture(GL_TEXTURE_2D, vTexture);
  bindVAO();
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  unbindVAO();
  glBindTexture(GL_TEXTURE_2D, 0);
}

void LiveTracer::update(CameraBase *_cam) {
  vCam           = _cam;
  uint32_t lMode = getRenderMode();

  vBundle  = lMode & 0b010;
  vBVHView = lMode & 0b001;

  if (vBVHView) {
    TimePoint lNow = std::chrono::system_clock::now();
    if (lNow - vPercentileRecalc > std::chrono::milliseconds(1000)) {
      vPercentileRecalc = lNow;
      vMaxCount         = calcIntCountPercentile(vDeviceImage, vWidth, vHeight, 0.99f);
    }
  }
}

std::string LiveTracer::getRenderModeString() {
  std::string lRet = vBundle ? "BUNDLE -- " : "NORMAL -- ";
  lRet += vBVHView ? "MAX int count: " + std::to_string(vMaxCount) : "Diffuse";
  return lRet;
}
