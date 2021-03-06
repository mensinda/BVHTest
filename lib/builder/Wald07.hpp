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

#include "BuilderBase.hpp"

namespace BVHTest::builder {

class Wald07 final : public BuilderBase {
 protected:
  ITER split(ITER _begin, ITER _end, uint32_t _level) override;

 public:
  Wald07() = default;
  virtual ~Wald07();

  std::string getName() const override { return "wald07"; }
  std::string getDesc() const override { return "Wald et al. 2007 sweep based BVH builder"; }

  base::ErrorCode runImpl(base::State &_state) override;
};

} // namespace BVHTest::builder
