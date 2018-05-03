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
  _default.y = _j.value("x", _default.y);
  _default.z = _j.value("x", _default.z);
  return _default;
}

json vec3ToJson(vec3 const &_vec) { return json{{"x", _vec.x}, {"y", _vec.y}, {"z", _vec.z}}; }

void Camera::fromJSON(const json &_j) {
  json lCamera     = _j.value("camera", json::object());
  json lResolution = _j.value("resolution", json::object());

  vPos    = vec3FromJson(lCamera.value("pos", json::object()), vPos);
  vLookAt = vec3FromJson(lCamera.value("lookAt", json::object()), vLookAt);
  vUp     = vec3FromJson(lCamera.value("up", json::object()), vUp);
  vFOV    = lCamera.value("fov", vFOV);

  vWidth  = lResolution.value("width", vWidth);
  vHeight = lResolution.value("height", vHeight);
}

json Camera::toJSON() const {
  return json{
      {"camera", {{"pos", vec3ToJson(vPos)}, {"lookAt", vec3ToJson(vLookAt)}, {"up", vec3ToJson(vUp)}, {"fov", vFOV}}},
      {"resolution", {{"width", vWidth}, {"height", vHeight}}}};
}

mat4 Camera::getViewProjection() {
  float lApectRatio = static_cast<float>(vWidth) / static_cast<float>(vHeight);
  mat4  lProjection = perspective(radians(vFOV), lApectRatio, 0.01f, 10.0f);
  mat4  lView       = lookAt(vPos, vLookAt, vUp);
  return lProjection * lView;
}
