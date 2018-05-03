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
#include "MeshRenderer.hpp"
#include <glm/gtc/type_ptr.hpp>
#include <iostream>
#include <string>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::view;
using namespace BVHTest::base;

struct VBOData {
  Vertex vert;
  Vertex norm;
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
  oColor = normalize(vNorm);
}
)__GLSL__";


MeshRenderer::MeshRenderer(const Mesh &_mesh) {
  auto lLogger = getLogger();

  glGenVertexArrays(1, &vVAO);
  glGenBuffers(1, &vVBO);
  glGenBuffers(1, &vEBO);

  // Generate OpenGL VBO data
  std::vector<VBOData> lOGLData;
  lOGLData.resize(_mesh.vert.size());
  for (size_t i = 0; i < _mesh.vert.size(); ++i) {
    lOGLData[i].vert = _mesh.vert[i];
    lOGLData[i].norm = _mesh.norm[i];
  }

  glBindVertexArray(vVAO);
  glBindBuffer(GL_ARRAY_BUFFER, vVBO);
  glBufferData(GL_ARRAY_BUFFER, lOGLData.size() * sizeof(VBOData), lOGLData.data(), GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vEBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, _mesh.faces.size() * sizeof(Triangle), _mesh.faces.data(), GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(VBOData), (void *)0);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(VBOData), (void *)offsetof(VBOData, norm));

  vNumIndex = _mesh.faces.size() * 3;

  glBindVertexArray(0);

  // Compile shaders
  vVertexShader   = glCreateShader(GL_VERTEX_SHADER);
  vFragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
  vShaderProg     = glCreateProgram();

  glShaderSource(vVertexShader, 1, &gVertexShader, nullptr);
  glShaderSource(vFragmentShader, 1, &gFragmentShader, nullptr);

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

  vUniformLoc = glGetUniformLocation(vShaderProg, "uMVP");
}

MeshRenderer::~MeshRenderer() {
  glDeleteBuffers(1, &vVBO);
  glDeleteBuffers(1, &vEBO);
  glDeleteVertexArrays(1, &vVAO);

  glDeleteShader(vVertexShader);
  glDeleteShader(vFragmentShader);
  glDeleteProgram(vShaderProg);
}

void MeshRenderer::update(glm::mat4 _mvp) { glUniformMatrix4fv(vUniformLoc, 1, GL_FALSE, glm::value_ptr(_mvp)); }

void MeshRenderer::render() {
  glUseProgram(vShaderProg);
  glBindVertexArray(vVAO);
  glDrawElements(GL_TRIANGLES, vNumIndex, GL_UNSIGNED_INT, 0);
  glBindVertexArray(0);
}
