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

#include "Window.hpp"
#include <iostream>

using namespace std;
using namespace BVHTest;
using namespace BVHTest::view;

Window::~Window() { destroy(); }

void Window::resizeHandler(GLFWwindow *_win, int _x, int _y) {
  glViewport(0, 0, _x, _y);
  Window *lWinPTR = static_cast<Window *>(glfwGetWindowUserPointer(_win));

  lWinPTR->vWidth  = _x;
  lWinPTR->vHeight = _y;
}

void Window::mouseHandler(GLFWwindow *_win, double _x, double _y) {
  Window *lWinPTR = static_cast<Window *>(glfwGetWindowUserPointer(_win));

  if (isnan(lWinPTR->vMousePosX) || isnan(lWinPTR->vMousePosY)) {
    lWinPTR->vMousePosX = _x;
    lWinPTR->vMousePosY = _y;
  }

  lWinPTR->vOffsetX += _x - lWinPTR->vMousePosX;
  lWinPTR->vOffsetY += lWinPTR->vMousePosY - _y;
  lWinPTR->vMousePosX = _x;
  lWinPTR->vMousePosY = _y;
}

void Window::scrollHandler(GLFWwindow *_win, double, double _yoffset) {
  Window *lWinPTR = static_cast<Window *>(glfwGetWindowUserPointer(_win));
  lWinPTR->vOffsetScroll += _yoffset;
}

void Window::keyHandler(GLFWwindow *_win, int _key, int, int _action, int) {
  Window *lWinPTR = static_cast<Window *>(glfwGetWindowUserPointer(_win));
  if (_action != GLFW_PRESS) return;
  lWinPTR->vKeyCallback(_key);
}


bool Window::create(std::string _title, uint32_t _x, uint32_t _y) {
  if (vWindow) destroy();

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);

  vWindow = glfwCreateWindow(_x, _y, _title.c_str(), nullptr, nullptr);

  if (!vWindow) { return false; }

  glfwMakeContextCurrent(vWindow);
  glfwSetInputMode(vWindow, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
  glfwSetWindowPos(vWindow, 0, 0);

  glfwSetWindowUserPointer(vWindow, this);
  glfwSetFramebufferSizeCallback(vWindow, resizeHandler);
  glfwSetCursorPosCallback(vWindow, mouseHandler);
  glfwSetScrollCallback(vWindow, scrollHandler);
  glfwSetKeyCallback(vWindow, keyHandler);
  glfwSwapInterval(1);

  vWidth  = _x;
  vHeight = _y;
  glViewport(0, 0, _x, _y);

  return true;
}

void Window::destroy() {
  if (vWindow) glfwDestroyWindow(vWindow);
  vWindow = nullptr;
}

bool Window::pollAndSwap() {
  if (!vWindow) return true;

  glfwSwapBuffers(vWindow);
  glfwPollEvents();
  return glfwWindowShouldClose(vWindow) == GLFW_FALSE;
}

bool Window::isKeyPressed(int _key) {
  if (!vWindow) return false;

  return glfwGetKey(vWindow, _key) == GLFW_PRESS;
}

void Window::setWindowShouldClose() {
  if (!vWindow) return;

  glfwSetWindowShouldClose(vWindow, GLFW_TRUE);
}

Window::RES Window::getResolution() const { return {vWidth, vHeight}; }

Window::MouseOffset Window::getMouseOffset() {
  MouseOffset lOffset = {vOffsetX, vOffsetY};
  vOffsetX            = 0;
  vOffsetY            = 0;
  return lOffset;
}

double Window::getScrollOffset() {
  auto lOffset  = vOffsetScroll;
  vOffsetScroll = 0;
  return lOffset;
}
