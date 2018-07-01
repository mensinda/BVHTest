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

#include <base/BVH.hpp>
#include "cudaFN.hpp"
#include "bucketSelect.cu"
#include <iostream>

using namespace glm;
using namespace std;
using namespace BVHTest;
using namespace BVHTest::base;
using namespace BVHTest::cuda;
using namespace BucketSelect;

#define CUDA_RUN(call)                                                                                                 \
  lRes = call;                                                                                                         \
  if (lRes != cudaSuccess) {                                                                                           \
    cout << "CUDA ERROR (" << __FILE__ << ":" << __LINE__ << "): " << cudaGetErrorString(lRes) << endl;                \
    goto error;                                                                                                        \
  }

#define ALLOCATE(ptr, num, type) CUDA_RUN(cudaMalloc(ptr, num * sizeof(type)));
#define FREE(ptr)                                                                                                      \
  cudaFree(ptr);                                                                                                       \
  ptr = nullptr;

#define MEMCOPY_WARPPER(dest, src, num, type, dev) CUDA_RUN(cudaMemcpy(dest, src, num * sizeof(type), dev))
#define COPY_TO_GPU(dest, src, num, type) MEMCOPY_WARPPER(dest, src, num, type, cudaMemcpyHostToDevice)
#define COPY_TO_HOST(dest, src, num, type) MEMCOPY_WARPPER(dest, src, num, type, cudaMemcpyDeviceToHost)

extern "C" bool copyBVHToGPU(BVH *_bvh, CUDAMemoryBVHPointer *_ptr) {
  if (!_bvh || !_ptr) { return false; }

  size_t      lNumNodes = _bvh->size();
  BVHNode     lData     = _bvh->data();
  BVH         lTempBVH;
  cudaError_t lRes;

  _ptr->numNodes = lNumNodes;

  ALLOCATE(&_ptr->nodes.bbox, lNumNodes, AABB);
  ALLOCATE(&_ptr->nodes.parent, lNumNodes, uint32_t);
  ALLOCATE(&_ptr->nodes.numChildren, lNumNodes, uint32_t);
  ALLOCATE(&_ptr->nodes.left, lNumNodes, uint32_t);
  ALLOCATE(&_ptr->nodes.right, lNumNodes, uint32_t);
  ALLOCATE(&_ptr->nodes.isLeft, lNumNodes, uint8_t);
  ALLOCATE(&_ptr->nodes.level, lNumNodes, uint16_t);
  ALLOCATE(&_ptr->nodes.surfaceArea, lNumNodes, float);

  COPY_TO_GPU(_ptr->nodes.bbox, lData.bbox, lNumNodes, AABB);
  COPY_TO_GPU(_ptr->nodes.parent, lData.parent, lNumNodes, uint32_t);
  COPY_TO_GPU(_ptr->nodes.numChildren, lData.numChildren, lNumNodes, uint32_t);
  COPY_TO_GPU(_ptr->nodes.left, lData.left, lNumNodes, uint32_t);
  COPY_TO_GPU(_ptr->nodes.right, lData.right, lNumNodes, uint32_t);
  COPY_TO_GPU(_ptr->nodes.isLeft, lData.isLeft, lNumNodes, uint8_t);
  COPY_TO_GPU(_ptr->nodes.level, lData.level, lNumNodes, uint16_t);
  COPY_TO_GPU(_ptr->nodes.surfaceArea, lData.surfaceArea, lNumNodes, float);

  lTempBVH.setNewRoot(_bvh->root());
  lTempBVH.setMaxLevel(_bvh->maxLevel());
  lTempBVH.setMemory(_ptr->nodes, lNumNodes, lNumNodes);

  ALLOCATE(&_ptr->bvh, 1, BVH);
  COPY_TO_GPU(_ptr->bvh, &lTempBVH, 1, BVH);

  lTempBVH.setMemory(BVHNode(), 0, 0); // Avoid destructor segfault

  return true;

error:
  FREE(_ptr->nodes.bbox);
  FREE(_ptr->nodes.parent);
  FREE(_ptr->nodes.numChildren);
  FREE(_ptr->nodes.left);
  FREE(_ptr->nodes.right);
  FREE(_ptr->nodes.isLeft);
  FREE(_ptr->nodes.level);
  FREE(_ptr->nodes.surfaceArea);
  FREE(_ptr->bvh);

  lTempBVH.setMemory(BVHNode(), 0, 0); // Avoid destructor segfault

  _ptr->numNodes = 0;
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



extern "C" bool copyBVHToHost(CUDAMemoryBVHPointer *_bvh, base::BVH *_ptr) {
  if (!_bvh || !_bvh->bvh || !_bvh->nodes.bbox || !_ptr) { return false; }

  cudaError_t lRes;
  BVHNode     lData;
  uint32_t    lNumNodes = 0;

  _ptr->clear();

  COPY_TO_HOST(_ptr, _bvh->bvh, 1, BVH);
  lNumNodes = _ptr->size();

  lData.bbox        = static_cast<AABB *>(malloc(lNumNodes * sizeof(AABB)));
  lData.parent      = static_cast<uint32_t *>(malloc(lNumNodes * sizeof(uint32_t)));
  lData.numChildren = static_cast<uint32_t *>(malloc(lNumNodes * sizeof(uint32_t)));
  lData.left        = static_cast<uint32_t *>(malloc(lNumNodes * sizeof(uint32_t)));
  lData.right       = static_cast<uint32_t *>(malloc(lNumNodes * sizeof(uint32_t)));
  lData.isLeft      = static_cast<uint8_t *>(malloc(lNumNodes * sizeof(uint8_t)));
  lData.level       = static_cast<uint16_t *>(malloc(lNumNodes * sizeof(uint16_t)));
  lData.surfaceArea = static_cast<float *>(malloc(lNumNodes * sizeof(float)));

  COPY_TO_HOST(lData.bbox, _bvh->nodes.bbox, lNumNodes, AABB);
  COPY_TO_HOST(lData.parent, _bvh->nodes.parent, lNumNodes, uint32_t);
  COPY_TO_HOST(lData.numChildren, _bvh->nodes.numChildren, lNumNodes, uint32_t);
  COPY_TO_HOST(lData.left, _bvh->nodes.left, lNumNodes, uint32_t);
  COPY_TO_HOST(lData.right, _bvh->nodes.right, lNumNodes, uint32_t);
  COPY_TO_HOST(lData.isLeft, _bvh->nodes.isLeft, lNumNodes, uint8_t);
  COPY_TO_HOST(lData.level, _bvh->nodes.level, lNumNodes, uint16_t);
  COPY_TO_HOST(lData.surfaceArea, _bvh->nodes.surfaceArea, lNumNodes, float);

  _ptr->setMemory(lData, lNumNodes, lNumNodes);
  lData = BVHNode(); // set everything to nullptr --> free below does nothing

error:
  FREE(_bvh->nodes.bbox);
  FREE(_bvh->nodes.parent);
  FREE(_bvh->nodes.numChildren);
  FREE(_bvh->nodes.left);
  FREE(_bvh->nodes.right);
  FREE(_bvh->nodes.isLeft);
  FREE(_bvh->nodes.level);
  FREE(_bvh->nodes.surfaceArea);
  FREE(_bvh->bvh);

  free(lData.bbox);
  free(lData.parent);
  free(lData.numChildren);
  free(lData.left);
  free(lData.right);
  free(lData.isLeft);
  free(lData.level);
  free(lData.surfaceArea);

  _bvh->bvh      = nullptr;
  _bvh->nodes    = BVHNode();
  _bvh->numNodes = 0;

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
  if (lRes != cudaSuccess) { cout << "CUDA ERROR: " << cudaGetErrorString(lRes) << endl; }

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

// __global__ void findTopKthElementDevice(DeviceTensor<float, 1> _data, uint32_t k, float *_out) {
//   float topK = warpFindTopKthElement(_data, k).k;
//
//   if (threadIdx.x == 0) {
//     *_out = topK;
//   }
// }


template <typename T>
struct results_t {
  float time;
  T     val;
};

template <typename T>
void setupForTiming(cudaEvent_t &start, cudaEvent_t &stop /*, T **d_vec, T* h_vec, uint size*/, results_t<T> **result) {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  //   cudaMalloc(d_vec, size * sizeof(T));
  //   cudaMemcpy(*d_vec, h_vec, size * sizeof(T), cudaMemcpyHostToDevice);
  *result = (results_t<T> *)malloc(sizeof(results_t<T>));
}

template <typename T>
void wrapupForTiming(cudaEvent_t &start, cudaEvent_t &stop /*, T* d_vec*/, results_t<T> *result, float time, T value) {
  //   cudaFree(d_vec);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  result->val  = value;
  result->time = time;
  //   cudaDeviceSynchronize();
}

extern "C" float topKThElement(float *_data, uint32_t size, uint32_t k) {
  //   float *dResult = nullptr;
  //   float lResult = 0.0f;
  //
  //   cudaMalloc(&dResult, 1 * sizeof(float));
  //
  //   int dataSizes[] = { (int) _num };
  //
  //   findTopKthElementDevice<<<1, 32>>>(DeviceTensor<float, 1>(_data, dataSizes), _k, dResult);
  //
  //   cudaMemcpy(&lResult, dResult, 1 * sizeof(float), cudaMemcpyDeviceToHost);
  //   cudaFree(dResult);
  //   return lResult;

  cudaEvent_t       start, stop;
  float             time;
  results_t<float> *result;
  float             retFromSelect;
  float *           deviceVec = _data;
  cudaDeviceProp    dp;
  cudaGetDeviceProperties(&dp, 0);


  setupForTiming(start, stop /*, &deviceVec, hostVec, size*/, &result);

  cudaEventRecord(start, 0);

  retFromSelect = bucketSelectWrapper(deviceVec, size, k, dp.multiProcessorCount, dp.maxThreadsPerBlock);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);


  wrapupForTiming(start, stop /*, deviceVec*/, result, time, retFromSelect);
  //   return result;

  return retFromSelect;
}



template <typename T>
void setupForTimingH(cudaEvent_t &start, cudaEvent_t &stop, T **d_vec, T *h_vec, uint size, results_t<T> **result) {
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaMalloc(d_vec, size * sizeof(T));
  cudaMemcpy(*d_vec, h_vec, size * sizeof(T), cudaMemcpyHostToDevice);
  *result = (results_t<T> *)malloc(sizeof(results_t<T>));
}

template <typename T>
void wrapupForTimingH(cudaEvent_t &start, cudaEvent_t &stop, T *d_vec, results_t<T> *result, float time, T value) {
  cudaFree(d_vec);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  result->val  = value;
  result->time = time;
  //   cudaDeviceSynchronize();
}

extern "C" float topKThElementHost(float *_data, uint32_t _num, uint32_t _k) {
  cudaEvent_t       start, stop;
  float             time;
  results_t<float> *result;
  float             retFromSelect;
  float *           deviceVec;
  cudaDeviceProp    dp;
  cudaGetDeviceProperties(&dp, 0);


  setupForTimingH(start, stop, &deviceVec, _data, _num, &result);

  cudaEventRecord(start, 0);

  retFromSelect = BucketSelect::bucketSelectWrapper(deviceVec, _num, _k, dp.multiProcessorCount, dp.maxThreadsPerBlock);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);


  wrapupForTimingH(start, stop, deviceVec, result, time, retFromSelect);
  return retFromSelect;
}
