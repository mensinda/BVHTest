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

#include "Configurable.hpp"
#include "State.hpp"

#define ENABLE_PROGRESS_BAR 1

#if ENABLE_PROGRESS_BAR
#  define PROGRESS(...) progress(__VA_ARGS__)
#  define PROGRESS_DONE progressDone()
#else
#  define PROGRESS(...)
#  define PROGRESS_DONE
#endif

namespace BVHTest::base {

enum class ErrorCode {
  OK,           // Everything went fine
  WARNING,      // Warning -- non fatal error
  IO_ERROR,     // Failed to read / write a file
  PARSE_ERROR,  // Paring a file failed
  BVH_ERROR,    // Something is wrong with the BVH
  GL_ERROR,     // OpenGL error
  CUDA_ERROR,   // CUDA error
  GENERIC_ERROR // Something went wrong
};

class Command : public Configurable {
 protected:
  virtual ErrorCode setup(State &) { return ErrorCode::OK; }
  virtual ErrorCode runImpl(State &_state) = 0;
  virtual void      teardown(State &) {}

  void progress(std::string _str, float _val);
  void progress(std::string _str, uint32_t _curr, uint32_t _max);
  void progressDone();

 public:
  Command() = default;
  virtual ~Command();

  ErrorCode run(State &_state);

  virtual std::string getName() const             = 0;
  virtual std::string getDesc() const             = 0;
  virtual CommandType getType() const             = 0;
  virtual uint64_t    getRequiredCommands() const = 0;
};

} // namespace BVHTest::base
