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

#define GLM_FORCE_NO_CTOR_INIT

#include <base/BVH.hpp>
#include <base/Ray.hpp>
#include <glm/vec3.hpp>
#include <cstdint>

using BVHTest::base::BVHNode;
using BVHTest::base::MeshRaw;
using BVHTest::base::Ray;
using glm::vec3;

struct CUDAPixel {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t intCount;
};

extern "C" bool allocateRays(Ray **_rays, uint32_t _num);
extern "C" bool allocateImage(uint8_t **_img, uint32_t _w, uint32_t _h);
extern "C" void freeRays(Ray **_rays);
extern "C" void freeImage(uint8_t *_img);

extern "C" void generateRays(Ray *_rays, uint32_t _w, uint32_t _h, vec3 _pos, vec3 _lookAt, vec3 _up, float _fov);
extern "C" void tracerImage(
    Ray *_rays, uint8_t *_img, BVHNode _nodes, uint32_t _rootNode, MeshRaw _mesh, vec3 _light, uint32_t _w, uint32_t _h);

extern "C" void copyImageToHost(CUDAPixel *_hostPixel, uint8_t *_cudaImg, uint32_t _w, uint32_t _h);

extern "C" void tracerDoCudaSync();
extern "C" void initCUDA_GL();
