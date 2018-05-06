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

#pragma once

#include "base/Command.hpp"
#include "RendererBase.hpp"
#include "Window.hpp"

namespace BVHTest::camera {
class Camera;
}

namespace BVHTest::view {

class Viewer final : public base::Command {
 private:
  uint32_t vResX             = 1920;
  uint32_t vResY             = 1080;
  double   vCamSpeed         = 0.00025;
  double   vMouseSensitivity = 0.025;
  bool     vFaceCulling      = false;

  struct RenderState {
    double vYaw   = -90;
    double vPitch = 0;

    bool vOverlay    = true;
    int  vSpeedLevel = 10;

    uint32_t vCurrCam = UINT32_MAX;

    Renderer vRendererType = Renderer::MESH;
  } vRState;

  std::shared_ptr<RendererBase> vRenderer = nullptr;

  struct ClearColor {
    float r = .47656250f;
    float g = .65625000f;
    float b = .89843750f;
    float a = 1.00f;
  } vClearColor;

  class GLFWInitHelper {
   private:
    bool isInit = false;

   public:
    GLFWInitHelper();
    ~GLFWInitHelper();
    inline bool ok() const { return isInit; }
  } vGLFWInit;

  void processInput(Window &_win, camera::Camera &_cam, uint32_t _time);
  void keyCallback(Window &_win, base::State &_state, camera::Camera &_cam, int _key);

  bool checkSetup(Window &_win, base::State &_state);
  void oglSetup();

 public:
  Viewer() = default;
  virtual ~Viewer();

  inline std::string       getName() const override { return "viewer"; }
  inline std::string       getDesc() const override { return "render the mesh, BVH, etc. with OpenGL"; }
  inline base::CommandType getType() const override { return base::CommandType::VIEWER; }
  inline uint64_t getRequiredCommands() const override { return static_cast<uint64_t>(base::CommandType::IMPORT); }

  base::ErrorCode runImpl(base::State &_state) override;

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::view
