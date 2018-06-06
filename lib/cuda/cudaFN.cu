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

#include "cudaFN.hpp"
#include <iostream>

using namespace glm;
using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::cuda;

#define CUDA_RUN(call)                                                                                                 \
  lRes = call;                                                                                                         \
  if (lRes != cudaSuccess) { goto error; }

extern "C" bool copyBVHToGPU(BVH *_bvh, BVH **_ptr) {
  if (!_bvh) { return false; }

  BVHNode *   lData = nullptr;
  size_t      lSize = _bvh->size() * sizeof(BVHNode);
  BVH         lTempBVH;
  cudaError_t lRes;

  CUDA_RUN(cudaMalloc(&lData, lSize));
  CUDA_RUN(cudaMemcpy(lData, _bvh->data(), lSize, cudaMemcpyHostToDevice));

  lTempBVH.setNewRoot(_bvh->root());
  lTempBVH.setMaxLevel(_bvh->maxLevel());
  lTempBVH.setMemory(lData, _bvh->size(), _bvh->size());

  CUDA_RUN(cudaMalloc(_ptr, sizeof(BVH)));
  CUDA_RUN(cudaMemcpy(*_ptr, &lTempBVH, sizeof(BVH), cudaMemcpyHostToDevice));

  lTempBVH.setMemory(nullptr, 0, 0); // Avoid destructor segfault

  return true;

error:
  if (lData) { cudaFree(lData); }
  if (*_ptr) { cudaFree(*_ptr); }

  lTempBVH.setMemory(nullptr, 0, 0); // Avoid destructor segfault

  *_ptr = nullptr;
  return false;
}

extern "C" bool copyMeshToGPU(Mesh *_mesh, MeshRaw *_meshOut) {
  if (!_mesh) { return false; }

  cudaError_t lRes;
  uint32_t    lVertSize  = _mesh->vert.size() * sizeof(vec3);
  uint32_t    lNromSize  = _mesh->norm.size() * sizeof(vec3);
  uint32_t    lFacesSize = _mesh->faces.size() * sizeof(Triangle);

  CUDA_RUN(cudaMalloc(&_meshOut->vert, lVertSize));
  CUDA_RUN(cudaMalloc(&_meshOut->norm, lNromSize));
  CUDA_RUN(cudaMalloc(&_meshOut->faces, lFacesSize));

  CUDA_RUN(cudaMemcpy(_meshOut->vert, _mesh->vert.data(), lVertSize, cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemcpy(_meshOut->norm, _mesh->norm.data(), lNromSize, cudaMemcpyHostToDevice));
  CUDA_RUN(cudaMemcpy(_meshOut->faces, _mesh->faces.data(), lFacesSize, cudaMemcpyHostToDevice));

  _meshOut->numVert  = _mesh->vert.size();
  _meshOut->numNorm  = _mesh->norm.size();
  _meshOut->numFaces = _mesh->faces.size();

  return true;

error:
  if (_meshOut->vert) { cudaFree(_meshOut->vert); }
  if (_meshOut->norm) { cudaFree(_meshOut->norm); }
  if (_meshOut->faces) { cudaFree(_meshOut->faces); }

  _meshOut->vert  = nullptr;
  _meshOut->norm  = nullptr;
  _meshOut->faces = nullptr;
  return false;
}



extern "C" bool copyBVHToHost(base::BVH **_bvh, base::BVH *_ptr) {
  if (!_bvh || !*_bvh || !_ptr) { return false; }

  cudaError_t lRes;
  BVH         lTempBVH;
  uint32_t    lSize   = 0;
  BVHNode *   lSource = nullptr;

  CUDA_RUN(cudaMemcpy(&lTempBVH, *_bvh, sizeof(BVH), cudaMemcpyDeviceToHost));

  _ptr->setNewRoot(lTempBVH.root());
  _ptr->setMaxLevel(lTempBVH.maxLevel());
  _ptr->resize(lTempBVH.size());

  lSize   = lTempBVH.size() * sizeof(BVHNode);
  lSource = lTempBVH.data();

  CUDA_RUN(cudaMemcpy(_ptr->data(), lSource, lSize, cudaMemcpyDeviceToHost));

error:
  cudaFree(_ptr);
  cudaFree(lSource);

  lTempBVH.setMemory(nullptr, 0, 0); // Avoid destructor segfault
  *_bvh = nullptr;

  return lRes == cudaSuccess;
}


extern "C" bool copyMeshToHost(base::MeshRaw *_mesh, base::Mesh *_meshOut) {
  if (!_mesh || !_mesh->vert || !_mesh->norm || !_mesh->faces || !_meshOut) { return false; }

  cudaError_t lRes;

  _meshOut->vert.resize(_mesh->numVert);
  _meshOut->norm.resize(_mesh->numNorm);
  _meshOut->faces.resize(_mesh->numFaces);

  uint32_t lVertSize  = _mesh->numVert * sizeof(vec3);
  uint32_t lNromSize  = _mesh->numNorm * sizeof(vec3);
  uint32_t lFacesSize = _mesh->numFaces * sizeof(Triangle);

  CUDA_RUN(cudaMemcpy(_meshOut->vert.data(), _mesh->vert, lVertSize, cudaMemcpyDeviceToHost));
  CUDA_RUN(cudaMemcpy(_meshOut->norm.data(), _mesh->norm, lNromSize, cudaMemcpyDeviceToHost));
  CUDA_RUN(cudaMemcpy(_meshOut->faces.data(), _mesh->faces, lFacesSize, cudaMemcpyDeviceToHost));

error:
  cudaFree(_mesh->vert);
  cudaFree(_mesh->norm);
  cudaFree(_mesh->faces);

  _mesh->vert  = nullptr;
  _mesh->norm  = nullptr;
  _mesh->faces = nullptr;

  _mesh->numVert  = 0;
  _mesh->numNorm  = 0;
  _mesh->numFaces = 0;

  return lRes == cudaSuccess;
}
