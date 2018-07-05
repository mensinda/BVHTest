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
  uint8_t  hit;
  uint8_t  diffuse;
  uint16_t intCount;
};

extern "C" bool allocateRays(Ray **_rays, uint32_t _num);
extern "C" bool allocateImage(CUDAPixel **_img, uint32_t _w, uint32_t _h);
extern "C" void freeRays(Ray **_rays);
extern "C" void freeImage(CUDAPixel *_img);

extern "C" void generateRays(Ray *_rays, uint32_t _w, uint32_t _h, vec3 _pos, vec3 _lookAt, vec3 _up, float _fov);
extern "C" void tracerImage(Ray *      _rays,
                            CUDAPixel *_img,
                            BVHNode *  _nodes,
                            uint32_t   _rootNode,
                            MeshRaw    _mesh,
                            vec3       _light,
                            uint32_t   _w,
                            uint32_t   _h,
                            bool       _bundle);

extern "C" void copyImageToHost(CUDAPixel *_hostPixel, CUDAPixel *_cudaImg, uint32_t _w, uint32_t _h);

extern "C" bool registerOGLImage(void **_resource, uint32_t _image);
extern "C" bool copyToOGLImage(void **_resource, CUDAPixel *_img, uint32_t _w, uint32_t _h);
extern "C" void unregisterOGLImage(void *_resource);

extern "C" uint16_t calcIntCountPercentile(CUDAPixel *_devImage, uint32_t _w, uint32_t _h, float _percent);

extern "C" void tracerDoCudaSync();
extern "C" void initCUDA_GL();
