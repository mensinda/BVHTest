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
#include <glm/mat4x4.hpp>
#include <glm/vec3.hpp>

namespace BVHTest::misc {

class Camera : public base::CameraBase {
 public:
  struct CAMERA {
    glm::vec3 pos    = glm::vec3(0.0f, 0.0f, 2.0f);
    glm::vec3 lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
    glm::vec3 up     = glm::vec3(0.0f, 1.0f, 0.0f);
    float     fov    = 45.0f;
  };

 private:
  CAMERA vCam;
  RES    vRes;

 public:
  Camera() = default;
  virtual ~Camera();

  inline void setFOV(float _fov) { vCam.fov = _fov; }
  inline void setResolution(uint32_t _width, uint32_t _height) {
    vRes.width  = _width;
    vRes.height = _height;
  }

  inline void setCamera(glm::vec3 _pos, glm::vec3 _lookAt, glm::vec3 _up) {
    vCam.pos    = _pos;
    vCam.lookAt = _lookAt;
    vCam.up     = _up;
  }

  inline RES    getResolution() const override { return vRes; }
  inline CAMERA getCamera() const { return vCam; }

  inline base::CameraType getType() const override { return base::CameraType::PERSPECTIVE; }

  std::vector<base::Ray> genRays() override;

  glm::mat4 getViewProjection();

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::misc
