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

#include "gl3w.h"

#include "BVHTestCfg.hpp"
#include "Viewer.hpp"

#include "MeshRenderer.hpp"
#include "Window.hpp"
#include <GLFW/glfw3.h>
#include <chrono>

using namespace std;
using namespace std::chrono;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::view;
using namespace BVHTest::camera;

Viewer::GLFWInitHelper::GLFWInitHelper() { isInit = glfwInit() == GLFW_TRUE && gl3wInit() != 0; }
Viewer::GLFWInitHelper::~GLFWInitHelper() { glfwTerminate(); }

Viewer::~Viewer() {}

void Viewer::fromJSON(const json &_j) {
  vResX             = _j.value("resX", vResX);
  vResY             = _j.value("resY", vResY);
  vFaceCulling      = _j.value("faceCulling", vFaceCulling);
  vWireframe        = _j.value("wireframe", vWireframe);
  vCamSpeed         = _j.value("camSpeed", vCamSpeed);
  vMouseSensitivity = _j.value("mouseSensitivity", vMouseSensitivity);

  json tmp = _j; // Make a non const copy
  if (tmp.count("clearColor") == 0) { tmp["clearColor"] = json::object(); }

  vClearColor.r = tmp["clearColor"].value("r", vClearColor.r);
  vClearColor.g = tmp["clearColor"].value("g", vClearColor.g);
  vClearColor.b = tmp["clearColor"].value("b", vClearColor.b);
  vClearColor.a = tmp["clearColor"].value("a", vClearColor.a);
}

json Viewer::toJSON() const {
  return json{{"resX", vResX},
              {"resY", vResY},
              {"faceCulling", vFaceCulling},
              {"wireframe", vWireframe},
              {"camSpeed", vCamSpeed},
              {"mouseSensitivity", vMouseSensitivity},
              {"clearColor", {{"r", vClearColor.r}, {"g", vClearColor.g}, {"b", vClearColor.b}, {"a", vClearColor.a}}}};
}

void Viewer::processInput(Window &_win, Camera &_cam, uint32_t _time) {
  if (_win.isKeyPressed(GLFW_KEY_ESCAPE)) _win.setWindowShouldClose();

  double lScroll = _win.getScrollOffset();
  auto [lX, lY]  = _win.getMouseOffset();

  // Get current stats
  auto [lPos, lLookAt, lUp] = _cam.getCamera();
  float lFOV                = _cam.getFOV();
  float lDelta              = _time * static_cast<float>(vCamSpeed);

  // Scrolling
  lFOV += static_cast<float>(lScroll);
  if (lFOV < 1.0) lFOV = 1.0;
  if (lFOV > 50.0) lFOV = 50.0;

  // Camera
  vYaw += lX * vMouseSensitivity;
  vPitch += lY * vMouseSensitivity;

  if (vPitch > 89.0) vPitch = 89.0;
  if (vPitch < -89.0) vPitch = -89.0;

  glm::vec3 lFront;
  lFront.x = cos(glm::radians(vYaw)) * cos(glm::radians(vPitch));
  lFront.y = sin(glm::radians(vPitch));
  lFront.z = sin(glm::radians(vYaw)) * cos(glm::radians(vPitch));
  lFront   = glm::normalize(lFront);

  // Movement (WASD)
  if (_win.isKeyPressed(GLFW_KEY_W)) lPos += lDelta * lFront;
  if (_win.isKeyPressed(GLFW_KEY_S)) lPos -= lDelta * lFront;
  if (_win.isKeyPressed(GLFW_KEY_A)) lPos -= lDelta * normalize(cross(lFront, lUp));
  if (_win.isKeyPressed(GLFW_KEY_D)) lPos += lDelta * normalize(cross(lFront, lUp));
  if (_win.isKeyPressed(GLFW_KEY_Q)) vRoll -= lDelta * 50;
  if (_win.isKeyPressed(GLFW_KEY_E)) vRoll += lDelta * 50;
  if (_win.isKeyPressed(GLFW_KEY_LEFT_SHIFT)) lPos += lDelta * lUp;
  if (_win.isKeyPressed(GLFW_KEY_LEFT_CONTROL)) lPos -= lDelta * lUp;

  lUp.x = cos(glm::radians(vRoll));
  lUp.y = sin(glm::radians(vRoll));
  lUp.z = 0;

  lLookAt = lPos + lFront;
  _cam.setCamera(lPos, lLookAt, lUp);
  _cam.setFOV(lFOV);
}


// Main loop
ErrorCode Viewer::runImpl(State &_state) {
  auto   lLogger = getLogger();
  Window lWindow;

  vYaw   = -90;
  vPitch = 0;
  vRoll  = 90;

  if (!vGLFWInit.ok() || !lWindow.create(_state.input, vResX, vResY)) {
    lLogger->error("Failed to create Window");
    return ErrorCode::GL_ERROR;
  }

  if (_state.mesh.vert.size() != _state.mesh.norm.size()) {
    lLogger->error("Invalid mesh data");
    return ErrorCode::GL_ERROR;
  }

  MeshRenderer lMesh(_state.mesh);
  Camera       lCam;

  glClearColor(vClearColor.r, vClearColor.g, vClearColor.b, vClearColor.a);
  glCullFace(GL_BACK);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_CULL_FACE);
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  if (vFaceCulling) { glEnable(GL_CULL_FACE); }
  if (vWireframe) { glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); }

  auto lCurr = system_clock::now();

  // main loop
  while (lWindow.pollAndSwap()) {
    auto lLast              = lCurr;
    lCurr                   = system_clock::now();
    milliseconds lFrameTime = duration_cast<milliseconds>(lCurr - lLast);

    processInput(lWindow, lCam, lFrameTime.count());

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    lMesh.update(lCam.getViewProjection());
    lMesh.render();
  }

  return ErrorCode::OK;
}
