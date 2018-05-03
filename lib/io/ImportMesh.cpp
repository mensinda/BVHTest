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

#include "BVHTestCfg.hpp"
#include "ImportMesh.hpp"
#include <assimp/Importer.hpp>
#include <assimp/SceneCombiner.h>
#include <assimp/postprocess.h>
#include <assimp/scene.h>

using namespace BVHTest::IO;
using namespace BVHTest::base;
using namespace Assimp;

ImportMesh::~ImportMesh() {}

void ImportMesh::fromJSON(const json &_j) {
  vOptimize  = _j.value("optimize", vOptimize);
  vNormalize = _j.value("normalize", vNormalize);
}

json ImportMesh::toJSON() const { return json{{"optimize", vOptimize}, {"normalize", vNormalize}}; }

ErrorCode ImportMesh::runImpl(State &_state) {
  auto     lLogger = getLogger();
  fs::path lPath   = fs::absolute(_state.basePath) / _state.input;
  lPath            = fs::canonical(lPath);
  Importer lImp;

  unsigned int lFlags =
      aiProcess_Triangulate | aiProcess_OptimizeMeshes | aiProcess_OptimizeGraph | aiProcess_RemoveComponent |
      aiProcess_GenSmoothNormals; // aiProcess_FindDegenerates | aiProcess_SortByPType | aiProcess_JoinIdenticalVertices

  if (vOptimize) lFlags |= aiProcess_ImproveCacheLocality;
  if (vNormalize) {
    lFlags |= aiProcess_PreTransformVertices;
    lImp.SetPropertyInteger(AI_CONFIG_PP_PTV_NORMALIZE, 1);
  }

  lImp.SetPropertyInteger(AI_CONFIG_PP_SBP_REMOVE, aiPrimitiveType_LINE | aiPrimitiveType_POINT);
  lImp.SetPropertyInteger(AI_CONFIG_PP_RVC_FLAGS,
                          aiComponent_TANGENTS_AND_BITANGENTS | aiComponent_COLORS | aiComponent_TEXCOORDS |
                              aiComponent_BONEWEIGHTS | aiComponent_ANIMATIONS | aiComponent_TEXTURES |
                              aiComponent_LIGHTS | aiComponent_CAMERAS | aiComponent_MATERIALS);

  const aiScene *lScene = lImp.ReadFile(lPath.string(), lFlags);

  if (!lScene) {
    lLogger->error("Failed to load file {}", lPath.string());
    lLogger->error("{}", lImp.GetErrorString());
    return ErrorCode::IO_ERROR;
  }

  if (lScene->mNumMeshes != 1) {
    lLogger->warn("Scene has {} meshes but only one mesh is supported", lScene->mNumMeshes);
    return ErrorCode::PARSE_ERROR;
  }

  aiMesh *lMesh = lScene->mMeshes[0];
  if (lMesh->mPrimitiveTypes != aiPrimitiveType_TRIANGLE) {
    lLogger->error("Mesh has non triangles");
    return ErrorCode::PARSE_ERROR;
  }

  _state.mesh.vert.resize(lMesh->mNumVertices);
  _state.mesh.norm.resize(lMesh->mNumVertices);
  _state.mesh.faces.resize(lMesh->mNumFaces);

  for (size_t i = 0; i < lMesh->mNumVertices; ++i) {
    _state.mesh.vert[i].x = lMesh->mVertices[i].x;
    _state.mesh.vert[i].y = lMesh->mVertices[i].y;
    _state.mesh.vert[i].z = lMesh->mVertices[i].z;

    _state.mesh.norm[i].x = lMesh->mNormals[i].x;
    _state.mesh.norm[i].y = lMesh->mNormals[i].y;
    _state.mesh.norm[i].z = lMesh->mNormals[i].z;
  }

  for (size_t i = 0; i < lMesh->mNumFaces; ++i) {
    _state.mesh.faces[i].v1 = lMesh->mFaces[i].mIndices[0];
    _state.mesh.faces[i].v2 = lMesh->mFaces[i].mIndices[1];
    _state.mesh.faces[i].v3 = lMesh->mFaces[i].mIndices[2];
  }

  return ErrorCode::OK;
}
