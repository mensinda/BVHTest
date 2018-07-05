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
#include "RendererBase.hpp"
#include "tracer/CUDAKernels.hpp"

using namespace glm;
using namespace std;
using namespace BVHTest;
using namespace BVHTest::view;
using namespace BVHTest::base;

RendererBase::RendererBase() {
  glGenVertexArrays(1, &vVAO);
  glGenBuffers(1, &vVBO);
  glGenBuffers(1, &vNBO);
  glGenBuffers(1, &vEBO);

  vVertexShader   = glCreateShader(GL_VERTEX_SHADER);
  vFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  vShaderProg     = glCreateProgram();
}

RendererBase::~RendererBase() {
  if (vCUDABuffer) { unregisterOGL(vCUDABuffer); }

  glDeleteBuffers(1, &vVBO);
  glDeleteBuffers(1, &vNBO);
  glDeleteBuffers(1, &vEBO);
  glDeleteVertexArrays(1, &vVAO);

  glDeleteProgram(vShaderProg);
}

bool RendererBase::compileShaders(const char *_vert, const char *_frag) {
  auto lLogger = getLogger();

  // Compile shaders
  glShaderSource(vVertexShader, 1, &_vert, nullptr);
  glShaderSource(vFragmentShader, 1, &_frag, nullptr);

  glCompileShader(vVertexShader);
  glCompileShader(vFragmentShader);

  // check for shader compile errors
  int  success;
  char infoLog[512];
  glGetShaderiv(vVertexShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vVertexShader, 512, NULL, infoLog);
    lLogger->error("Vertex shader compilation failed");
    lLogger->error("{}", infoLog);
  }

  glGetShaderiv(vFragmentShader, GL_COMPILE_STATUS, &success);
  if (!success) {
    glGetShaderInfoLog(vFragmentShader, 512, NULL, infoLog);
    lLogger->error("Fragment shader compilation failed");
    lLogger->error("{}", infoLog);
  }

  // link shaders
  glAttachShader(vShaderProg, vVertexShader);
  glAttachShader(vShaderProg, vFragmentShader);
  glLinkProgram(vShaderProg);

  // check for linking errors
  glGetProgramiv(vShaderProg, GL_LINK_STATUS, &success);
  if (!success) {
    glGetProgramInfoLog(vShaderProg, 512, NULL, infoLog);
    lLogger->error("Shader linking failed");
    lLogger->error("{}", infoLog);
  }

  glDetachShader(vShaderProg, vVertexShader);
  glDetachShader(vShaderProg, vFragmentShader);
  glDeleteShader(vVertexShader);
  glDeleteShader(vFragmentShader);
  return true;
}

GLint RendererBase::getLocation(const char *_name) { return glGetUniformLocation(vShaderProg, _name); }


bool RendererBase::generateVBOData(uint32_t _numVert) {
  bindVAO();
  bindVBO();
  glBufferData(GL_ARRAY_BUFFER, _numVert * sizeof(vec3), nullptr, GL_DYNAMIC_DRAW);
  glBindBuffer(GL_ARRAY_BUFFER, 0);

  return registerOGLBuffer(&vCUDABuffer, vVBO);
}

bool RendererBase::copyVBODataDevice2Device(vec3 *_data, uint32_t _size) {
  return copyToOGLBuffer(&vCUDABuffer, _data, _size);
}

void RendererBase::updateMesh(State &, CameraBase *, uint32_t) {}
