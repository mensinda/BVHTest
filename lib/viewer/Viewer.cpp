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

#include "camera/Camera.hpp"
#include <glm/glm.hpp>
#include "BVHRenderer.hpp"
#include "MeshRenderer.hpp"
#include <GLFW/glfw3.h>
#include <chrono>


#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wunused-function"
#define GLT_MANUAL_VIEWPORT
#include "gltext.h"
#pragma GCC diagnostic pop

using namespace std;
using namespace std::chrono;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::view;
using namespace BVHTest::camera;

class TextInit final {
 public:
  TextInit() { gltInit(); }
  ~TextInit() { gltTerminate(); }
};

class Text final {
 private:
  GLTtext *vText  = nullptr;
  GLfloat  vScale = 1;
  GLfloat  vPosX  = 0;
  GLfloat  vPosY  = 0;

 public:
  Text() { vText = gltCreateText(); }
  ~Text() { gltDeleteText(vText); }

  float lineHeight() { return gltGetLineHeight(vScale); }
  void  set(string _s) { gltSetText(vText, _s.c_str()); }
  void  setScale(GLfloat _scale) { vScale = _scale; }
  void  setPos(GLfloat _x, GLfloat _y) {
    vPosX = _x;
    vPosY = _y;
  }
  void setLine(int _l) { vPosY = lineHeight() * _l; }
  void draw() { gltDrawText2D(vText, vPosX, vPosY, vScale); }
};

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
  double lScroll = _win.getScrollOffset();
  auto [lX, lY]  = _win.getMouseOffset();

  if (lX != 0 || lY != 0 || lScroll != 0) { vRState.vCurrCam = UINT32_MAX; }
  if (vRState.vCurrCam != UINT32_MAX) { return; }

  // Get current stats
  auto [lPos, lLookAt, lUp] = _cam.getCamera();
  float lDelta              = _time * vCamSpeed * (static_cast<float>(vRState.vSpeedLevel) / 10.0f);
  float lFOV                = _cam.getFOV();

  // Scrolling
  lFOV += static_cast<float>(lScroll);
  if (lFOV < 1.0) lFOV = 1.0;
  if (lFOV > 50.0) lFOV = 50.0;

  // Camera
  vRState.vYaw += lX * vMouseSensitivity;
  vRState.vPitch += lY * vMouseSensitivity;

  if (vRState.vPitch > 89.0) vRState.vPitch = 89.0;
  if (vRState.vPitch < -89.0) vRState.vPitch = -89.0;


  glm::vec3 lFront;
  lFront.x = cos(glm::radians(vRState.vYaw)) * cos(glm::radians(vRState.vPitch));
  lFront.y = sin(glm::radians(vRState.vPitch));
  lFront.z = sin(glm::radians(vRState.vYaw)) * cos(glm::radians(vRState.vPitch));

  // Movement (WASD)
  if (_win.isKeyPressed(GLFW_KEY_W)) lPos += lDelta * lFront;
  if (_win.isKeyPressed(GLFW_KEY_S)) lPos -= lDelta * lFront;
  if (_win.isKeyPressed(GLFW_KEY_A)) lPos -= lDelta * normalize(cross(lFront, lUp));
  if (_win.isKeyPressed(GLFW_KEY_D)) lPos += lDelta * normalize(cross(lFront, lUp));
  if (_win.isKeyPressed(GLFW_KEY_LEFT_SHIFT)) lPos += lDelta * lUp;
  if (_win.isKeyPressed(GLFW_KEY_LEFT_CONTROL)) lPos -= lDelta * lUp;

  lLookAt = lPos + lFront;
  _cam.setCamera(lPos, lLookAt, lUp);
  _cam.setFOV(lFOV);
}

void Viewer::keyCallback(Window &_win, State &_state, Camera &_cam, int _key) {
  switch (_key) {
    case GLFW_KEY_ESCAPE: _win.setWindowShouldClose(); break;
    case GLFW_KEY_KP_ADD: vRState.vSpeedLevel++; break;
    case GLFW_KEY_KP_SUBTRACT: vRState.vSpeedLevel--; break;
    case GLFW_KEY_BACKSPACE: vRState.vOverlay = !vRState.vOverlay; break;
    case GLFW_KEY_ENTER: _state.cameras.push_back(make_shared<Camera>(_cam)); break;
    case GLFW_KEY_C: {
      if (_state.cameras.empty()) { break; }

      vRState.vCurrCam++; // Undefined is UINT32_MAX --> UINT32_MAX + 1 == 0 (overflow by design!)
      if (vRState.vCurrCam != UINT32_MAX) {
        if (vRState.vCurrCam >= _state.cameras.size()) { vRState.vCurrCam = 0; }
      }

      Camera *lCamSaved = dynamic_cast<Camera *>(_state.cameras[vRState.vCurrCam].get());
      if (!lCamSaved) break;

      _cam           = *lCamSaved;
      vRState.vPitch = 0;
      vRState.vYaw   = -90;
      break;
    }
    case GLFW_KEY_DELETE:
      if (vRState.vCurrCam == UINT32_MAX) { break; }
      _state.cameras.erase(_state.cameras.begin() + vRState.vCurrCam);
      vRState.vCurrCam = UINT32_MAX;
      break;

    case GLFW_KEY_W:
    case GLFW_KEY_A:
    case GLFW_KEY_S:
    case GLFW_KEY_D: vRState.vCurrCam = UINT32_MAX; break;
  }

  if (vRState.vSpeedLevel < 1) vRState.vSpeedLevel = 1;
  if (vRState.vSpeedLevel > 19) vRState.vSpeedLevel = 19;
}


bool Viewer::checkSetup(Window &_win, State &_state) {
  auto lLogger = getLogger();
  if (!vGLFWInit.ok() || !_win.getIsCreated()) {
    lLogger->error("Failed to create Window");
    return false;
  }

  if (_state.mesh.vert.size() != _state.mesh.norm.size()) {
    lLogger->error("Invalid mesh data");
    return false;
  }

  return true;
}

void Viewer::oglSetup() {
  glClearColor(vClearColor.r, vClearColor.g, vClearColor.b, vClearColor.a);
  glCullFace(GL_BACK);
  glEnable(GL_DEPTH_TEST);
  glEnable(GL_BLEND);
  glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

  if (vFaceCulling) { glEnable(GL_CULL_FACE); }
  if (vWireframe) { glPolygonMode(GL_FRONT_AND_BACK, GL_LINE); }

  gltInit();
  gltColor(1.0f, 1.0f, 1.0f, 1.0f);
}



// Main loop
ErrorCode Viewer::runImpl(State &_state) {
  vRState = RenderState(); // Reset internal state

  Window lWindow;
  lWindow.create(_state.input, vResX, vResY);
  if (!checkSetup(lWindow, _state)) return ErrorCode::GL_ERROR;

  TextInit lTextInit;
  Camera   lCam;
  Text     lFPSText;
  Text     lControl;
  Text     lUsage;
  uint32_t lFrames  = 0;
  uint32_t lLastFPS = 0;
  uint32_t lFPS     = 0;

  lWindow.setKeyCallback([&](int _key) -> void { keyCallback(lWindow, _state, lCam, _key); });

  oglSetup();

  lFPSText.setLine(0);
  lControl.setLine(1);

  lUsage.set(
      "Movemet:   WASD + Mouse + Scroll wheel    ###   Movement Speed: KP +/-"
      "\nCameras:"
      "\n - Cycle: C"
      "\n - Add: ENTER"
      "\n - Delete: DELETE"
      "\nToggle overlay: BACKSPACE");

  auto lCurr         = high_resolution_clock::now();
  auto lFPSTimeStamp = high_resolution_clock::now();

  // main loop
  while (lWindow.pollAndSwap()) {
    auto lLast          = lCurr;
    lCurr               = high_resolution_clock::now();
    uint32_t lFrameTime = duration_cast<milliseconds>(lCurr - lLast).count();

    processInput(lWindow, lCam, lFrameTime);

    if (!vRenderer || vRState.vRendererType != vRenderer->getType()) {
      switch (vRState.vRendererType) {
        case Renderer::MESH: vRenderer = make_shared<MeshRenderer>(_state.mesh); break;
        case Renderer::BVH: vRenderer = make_shared<BVHRenderer>(_state.aabbs); break;
      }
    }

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    vRenderer->update(&lCam);
    vRenderer->render();

    if (vRState.vOverlay) {
      auto [width, height] = lWindow.getResolution();
      if (duration_cast<milliseconds>(high_resolution_clock::now() - lFPSTimeStamp) > milliseconds(500)) {
        lFPSTimeStamp = high_resolution_clock::now();
        lFPS          = (lFrames - lLastFPS - 1) * 2;
        lLastFPS      = lFrames;
      }

      gltViewport(width, height);
      lFPSText.set(fmt::format("FPS: {}; Frametime: {}ms", lFPS, lFrameTime));
      lControl.set(fmt::format("Speed level: {}\nSaved cameras: {}\nCurrent camera: {}",
                               vRState.vSpeedLevel,
                               _state.cameras.size(),
                               vRState.vCurrCam == UINT32_MAX ? "-" : to_string(vRState.vCurrCam)));
      lUsage.setPos(0, static_cast<float>(height) - lUsage.lineHeight() * 6);
      lFPSText.draw();
      lControl.draw();
      lUsage.draw();
    }

    lFrames++;
  }

  vRenderer = nullptr;
  gltTerminate();
  return ErrorCode::OK;
}
