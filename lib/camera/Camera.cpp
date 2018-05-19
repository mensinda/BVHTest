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

#include "Camera.hpp"
#include <glm/gtc/matrix_transform.hpp>

using namespace std;
using namespace glm;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::camera;

Camera::~Camera() {}

vec3 vec3FromJson(const json &_j, vec3 _default) {
  _default.x = _j.value("x", _default.x);
  _default.y = _j.value("y", _default.y);
  _default.z = _j.value("z", _default.z);
  return _default;
}

json vec3ToJson(vec3 const &_vec) { return json{{"x", _vec.x}, {"y", _vec.y}, {"z", _vec.z}}; }

void Camera::fromJSON(const json &_j) {
  json lCamera     = _j.value("camera", json::object());
  json lResolution = _j.value("resolution", json::object());

  vCam.pos    = vec3FromJson(lCamera.value("pos", json::object()), vCam.pos);
  vCam.lookAt = vec3FromJson(lCamera.value("lookAt", json::object()), vCam.lookAt);
  vCam.up     = vec3FromJson(lCamera.value("up", json::object()), vCam.up);
  vCam.fov    = lCamera.value("fov", vCam.fov);

  vRes.width  = lResolution.value("width", vRes.width);
  vRes.height = lResolution.value("height", vRes.height);
}

json Camera::toJSON() const {
  return json{{"camera",
               {{"pos", vec3ToJson(vCam.pos)},
                {"lookAt", vec3ToJson(vCam.lookAt)},
                {"up", vec3ToJson(vCam.up)},
                {"fov", vCam.fov}}},
              {"resolution", {{"width", vRes.width}, {"height", vRes.height}}}};
}

mat4 Camera::getViewProjection() {
  float lApectRatio = static_cast<float>(vRes.width) / static_cast<float>(vRes.height);
  mat4  lProjection = perspective(radians(vCam.fov), lApectRatio, 0.01f, 10.0f);
  mat4  lView       = lookAt(vCam.pos, vCam.lookAt, vCam.up);
  return lProjection * lView;
}


vector<Ray> Camera::genRays() {
  vector<Ray> lRays;

  lRays.resize(vRes.width * vRes.height);
  mat4  lCamToWorld  = inverse(lookAtRH(vCam.pos, vCam.lookAt, vCam.up));
  vec3  lOrigin      = vCam.pos;
  float lAspectRatio = static_cast<float>(vRes.width) / static_cast<float>(vRes.height);
  float lScale       = tan(radians(0.5 * vCam.fov));

#pragma omp parallel for collapse(2)
  for (uint32_t y = 0; y < vRes.height; ++y) {
    for (uint32_t x = 0; x < vRes.width; ++x) {
      float lPixX      = (2 * ((x + 0.5) / vRes.width) - 1) * lScale * lAspectRatio;
      float lPixY      = (1 - 2 * ((y + 0.5) / vRes.height)) * lScale;
      vec3  lDirection = lCamToWorld * vec4(lPixX, lPixY, -1, 0.0f);

      lRays[y * vRes.width + x] = Ray(lOrigin, normalize(lDirection), x, y);
    }
  }

  return lRays;
}
