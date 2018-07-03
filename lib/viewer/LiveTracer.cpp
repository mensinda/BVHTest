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

bool LiveTracer::vCudaIsInit = false;

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

uniform sampler2D uTex;

void main() {
  oColor = vec4(vec3(texture(uTex, vTexCoord)), 1.0f);
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
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, vWidth, vHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glBindTexture(GL_TEXTURE_2D, 0);

  if (!compileShaders(gVertexShader, gFragmentShader)) { return; }

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
  tracerImage(vRays, vDeviceImage, vCudaMem.bvh.nodes, 0, vCudaMem.rawMesh, vec3(1, 1, 1), vWidth, vHeight, true);
  copyToOGLImage(&vCudaRes, vDeviceImage, vWidth, vHeight);

  useProg();
  glBindTexture(GL_TEXTURE_2D, vTexture);
  bindVAO();
  glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
  unbindVAO();
  glBindTexture(GL_TEXTURE_2D, 0);
}

void LiveTracer::update(CameraBase *_cam) { vCam = _cam; }

bool LiveTracer::cudaInit() {
  if (!vCudaIsInit) {
    //     initCUDA_GL();
    vCudaIsInit = true;
  }
  return vCudaIsInit;
}
