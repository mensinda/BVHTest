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

#include "base/CameraBase.hpp"
#include <glm/glm.hpp>

namespace BVHTest::camera {

class Camera : public base::CameraBase {
 public:
  struct RES {
    uint32_t width;
    uint32_t height;
  };

  struct CAMERA {
    glm::vec3 pos;
    glm::vec3 lookAt;
    glm::vec3 up;
  };

 private:
  glm::vec3 vPos    = glm::vec3(0.0f, 0.0f, 2.0f);
  glm::vec3 vLookAt = glm::vec3(0.0f, 0.0f, 0.0f);
  glm::vec3 vUp     = glm::vec3(0.0f, 1.0f, 0.0f);

  float    vFOV    = 45.0f;
  uint32_t vWidth  = 1920;
  uint32_t vHeight = 1080;

 public:
  Camera() = default;
  virtual ~Camera();

  inline void setFOV(float _fov) { vFOV = _fov; }
  inline void setResolution(uint32_t _width, uint32_t _height) {
    vWidth  = _width;
    vHeight = _height;
  }

  inline void setCamera(glm::vec3 _pos, glm::vec3 _lookAt, glm::vec3 _up) {
    vPos    = _pos;
    vLookAt = _lookAt;
    vUp     = _up;
  }

  inline float  getFOV() const { return vFOV; }
  inline RES    getResolution() const { return {vWidth, vHeight}; }
  inline CAMERA getCamera() const { return {vPos, vLookAt, vUp}; }

  inline base::CameraType getType() const override { return base::CameraType::PERSPECTIVE; }

  glm::mat4 getViewProjection();

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::camera
