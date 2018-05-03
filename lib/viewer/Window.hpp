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

#include <GLFW/glfw3.h>
#include <cmath>
#include <string>

namespace BVHTest::view {

class Window {
 public:
  struct RES {
    uint32_t width;
    uint32_t height;
  };

  struct MouseOffset {
    double x;
    double y;
  };

 private:
  std::string vTitle;
  GLFWwindow *vWindow = nullptr;

  uint32_t vWidth  = 0;
  uint32_t vHeight = 0;

  double vMousePosX = std::nan("-");
  double vMousePosY = std::nan("-");

  double vOffsetX      = 0;
  double vOffsetY      = 0;
  double vOffsetScroll = 0;

  static void resizeHandler(GLFWwindow *_win, int _x, int _y);
  static void mouseHandler(GLFWwindow *_win, double _x, double _y);
  static void scrollHandler(GLFWwindow *_win, double _xoffset, double _yoffset);

 public:
  Window() = default;
  ~Window();

  bool create(std::string _title, uint32_t _x, uint32_t _y);
  void destroy();

  bool pollAndSwap();
  bool isKeyPressed(int _key);
  void setWindowShouldClose();

  inline bool getIsCreated() const { return vWindow != nullptr; }
  RES         getResolution() const;
  MouseOffset getMouseOffset();
  double      getScrollOffset();
};

} // namespace BVHTest::view
