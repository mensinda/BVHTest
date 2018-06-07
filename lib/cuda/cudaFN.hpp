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

namespace BVHTest {
namespace cuda {

extern "C" bool copyBVHToGPU(base::BVH *_bvh, base::CUDAMemoryBVHPointer *_ptr);
extern "C" bool copyMeshToGPU(base::Mesh *_mesh, base::MeshRaw *_meshOut);

extern "C" bool copyBVHToHost(base::CUDAMemoryBVHPointer *_bvh, base::BVH *_ptr);
extern "C" bool copyMeshToHost(base::MeshRaw *_mesh, base::Mesh *_meshOut);

} // namespace cuda
} // namespace BVHTest
