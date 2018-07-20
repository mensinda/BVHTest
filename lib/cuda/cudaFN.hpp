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

#include "base/BVH.hpp"
#include <glm/mat4x4.hpp>

enum class MemcpyKind { Host2Host = 0, Host2Dev = 1, Dev2Host = 2, Dev2Dev = 3, Default = 4 };

namespace BVHTest {
namespace cuda {

extern "C" bool copyBVHToGPU(base::BVH *_bvh, base::CUDAMemoryBVHPointer *_ptr);
extern "C" bool copyMeshToGPU(base::Mesh *_mesh, base::MeshRaw *_meshOut);

extern "C" bool copyBVHToHost(base::CUDAMemoryBVHPointer *_bvh, base::BVH *_ptr);
extern "C" bool copyMeshToHost(base::MeshRaw *_mesh, base::Mesh *_meshOut);

extern "C" void resetCUDA();

extern "C" float topKThElement(float *_data, uint32_t _num, uint32_t _k);
extern "C" float topKThElementHost(float *_data, uint32_t _num, uint32_t _k);

extern "C" bool runMalloc(void **_ptr, size_t _size);
extern "C" bool runMemcpy(void *_dest, void *_src, size_t _size, MemcpyKind _kind);
extern "C" void runFree(void *_ptr);

extern "C" void transformVecs(glm::vec3 *_src, glm::vec3 *_dest, uint32_t _size, glm::mat4 _mat);

} // namespace cuda
} // namespace BVHTest
