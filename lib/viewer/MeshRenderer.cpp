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

#define GLM_ENABLE_EXPERIMENTAL

#include "MeshRenderer.hpp"
#include "cuda/cudaFN.hpp"
#include "misc/Camera.hpp"
#include <glm/gtx/transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>

using namespace glm;
using namespace std;
using namespace BVHTest;
using namespace BVHTest::view;
using namespace BVHTest::base;
using namespace BVHTest::misc;

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


MeshRenderer::MeshRenderer(State &_state) {
  auto const &lMesh = _state.mesh;

  vRawMesh = _state.cudaMem.rawMesh;
  vNumVert = vRawMesh.numVert * sizeof(vec3);
  cuda::runMalloc((void **)&vDevOriginalVert, vNumVert);
  cuda::runMemcpy(vDevOriginalVert, vRawMesh.vert, vNumVert, MemcpyKind::Dev2Dev);

  bindVAO();
  generateVBOData(vRawMesh.numVert);
  copyVBODataDevice2Device(vRawMesh.vert, vRawMesh.numVert);

  bindNBO();
  glBufferData(GL_ARRAY_BUFFER, lMesh.norm.size() * sizeof(vec3), lMesh.norm.data(), GL_STATIC_DRAW);

  bindEBO();
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, lMesh.faces.size() * sizeof(Triangle), lMesh.faces.data(), GL_STATIC_DRAW);

  bindVBO();
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

  bindNBO();
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);

  unbindVAO();

  if (!compileShaders(gVertexShader, gFragmentShader)) { return; }

  vNumIndex   = lMesh.faces.size() * 3;
  vUniformLoc = getLocation("uMVP");
}

MeshRenderer::~MeshRenderer() {
  cuda::runMemcpy(vRawMesh.vert, vDevOriginalVert, vNumVert, MemcpyKind::Dev2Dev);
  cuda::runFree(vDevOriginalVert);
}

void MeshRenderer::update(CameraBase *_cam) {
  Camera *lCam = dynamic_cast<Camera *>(_cam);
  if (!lCam) { return; }

  useProg();
  glUniformMatrix4fv(vUniformLoc, 1, GL_FALSE, glm::value_ptr(lCam->getViewProjection()));

  if (getRenderMode() == 0) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  } else {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  }
}

void MeshRenderer::updateMesh(State &_state, CameraBase *_cam, uint32_t _offsetIndex) {
  Camera *lCam = dynamic_cast<Camera *>(_cam);
  if (!lCam) { return; }

  auto lData = lCam->getCamera();
  auto lOffs = _state.meshOffsets[_offsetIndex];

  mat4 lMat = translate(lData.pos);

  cuda::transformVecs(vDevOriginalVert + lOffs.vertOffset,
                      _state.cudaMem.rawMesh.vert + lOffs.vertOffset,
                      _state.cudaMem.rawMesh.numVert - lOffs.vertOffset,
                      lMat);
  copyVBODataDevice2Device(_state.cudaMem.rawMesh.vert, _state.cudaMem.rawMesh.numVert);
}


void MeshRenderer::render() {
  useProg();
  bindVAO();
  glDrawElements(GL_TRIANGLES, vNumIndex, GL_UNSIGNED_INT, 0);
  unbindVAO();
}

std::string MeshRenderer::getRenderModeString() { return getRenderMode() == 0 ? "Faces" : "Wireframe"; }
