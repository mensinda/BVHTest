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

#include "BVH.hpp"
#include "CameraBase.hpp"
#include "Image.hpp"
#include <chrono>
#include <memory>
#include <string>
#include <vector>

using std::chrono::milliseconds;
using std::chrono::nanoseconds;

namespace BVHTest::base {

enum class CommandType {
  IMPORT    = (1 << 0), // Data was imported
  BVH_BUILD = (1 << 4),
  BVH_OPT1  = (1 << 5),
  BVH_OPT2  = (1 << 6),
  CUDA_INIT = (1 << 8),
  CUDA_DONE = (1 << 7),
  RAY_TRACE = (1 << 16),
  EXPORT    = (1 << 24), // Data was exported
  SUMMARY   = (1 << 28),
  VIEWER    = (1 << 29) // Debug OGL viewer for the mesh, BVH, etc.
};


struct State final {
  typedef std::shared_ptr<CameraBase> CAM_PTR;

  uint64_t    commandsRun = 0;
  std::string basePath    = "";
  std::string input       = "";
  std::string name        = "";

  struct CMD final {
    std::string  name;
    CommandType  type;
    milliseconds duration;
  };

  struct Work final {
    std::vector<Ray> rays;
    Image            img  = {{}, 0, 0, "IMG"};
    nanoseconds      time = nanoseconds(0);
  };

  struct NewMeshOffset {
    uint32_t vertOffset;
    uint32_t normOffset;
    uint32_t facesOffset;
  };

  struct OptData {
    struct OptStepp {
      uint32_t     step;
      float        sah;
      milliseconds duration;
    };

    std::vector<OptStepp> optStepps;


    uint32_t numSkipped;
    uint32_t numTotal;
  } optData;

  struct CudaMemory final {
    CUDAMemoryBVHPointer bvh;
    MeshRaw              rawMesh;
  } cudaMem;

  std::vector<CMD> commands;

  Mesh                       mesh;
  std::vector<NewMeshOffset> meshOffsets;
  std::vector<CAM_PTR>       cameras;
  std::vector<CAM_PTR>       camTrac;
  std::vector<Work>          work;
  BVH                        bvh;
};

} // namespace BVHTest::base
