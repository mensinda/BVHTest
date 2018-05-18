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

namespace BVHTest::tracer {

struct Pixel {
  uint8_t  r;
  uint8_t  g;
  uint8_t  b;
  uint32_t intCount;
};

class TracerBase : public base::Command {
 private:
  glm::vec3 vLightLocation    = {2.0f, 2.0f, 2.0f};
  uint32_t  vMaxIntersections = 100;

 public:
  TracerBase() = default;
  virtual ~TracerBase();

  base::ErrorCode writeImage(
      std::vector<Pixel> const &_pixels, uint32_t _width, uint32_t _height, std::string _name, base::State &_state);

  inline base::CommandType getType() const override { return base::CommandType::RAY_TRACE; }
  inline uint64_t getRequiredCommands() const override { return static_cast<uint64_t>(base::CommandType::BVH_BUILD); }

  inline glm::vec3 getLightLocation() const { return vLightLocation; }

  void fromJSON(const json &_j) override;
  json toJSON() const override;
};

} // namespace BVHTest::tracer
