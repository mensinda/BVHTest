#################
#### Sources ####
#################

# Base lib
baseSrc = files([
  'base/BVH.cpp',
  'base/CameraBase.cpp',
  'base/Configurable.cpp',
  'base/Config.cpp',
  'base/Command.cpp',
  'base/Ray.cpp',
  'base/StatusDump.cpp'
])

baseInc = files([
  'base/BVH.hpp',
  'base/BVHPatch.hpp',
  'base/CameraBase.hpp',
  'base/Configurable.hpp',
  'base/Config.hpp',
  'base/Command.hpp',
  'base/Ray.hpp',
  'base/State.hpp',
  'base/StatusDump.hpp',
  'base/StringHash.hpp'
])


# Builder lib
builderSrc = files([
  'builder/Bittner13.cpp',
  'builder/Bittner13Par.cpp',
  'builder/Bittner13GPU.cpp',
  'builder/BuilderBase.cpp',
  'builder/HLBVH.cpp',
  'builder/Median.cpp',
  'builder/OptimizerBase.cpp',
  'builder/Wald07.cpp'
])


builderCU = files([
  'builder/Bittner13CUDA.cu',
  'builder/HLBVH_CUDA.cu'
])


builderInc = files([
  'builder/Bittner13.hpp',
  'builder/Bittner13Par.hpp',
  'builder/Bittner13GPU.hpp',
  'builder/BuilderBase.hpp',
  'builder/HLBVH.hpp',
  'builder/Median.hpp',
  'builder/OptimizerBase.hpp',
  'builder/Wald07.hpp'
])


# Camera lib
miscSrc = files([
  'misc/Camera.cpp',
  'misc/Sleep.cpp',
  'misc/Validate.cpp'
])

miscInc = files([
  'misc/Camera.hpp',
  'misc/Sleep.hpp',
  'misc/Validate.hpp'
])


# IO lib
ioSrc = files([
  'io/BVHExport.cpp',
  'io/BVHImport.cpp',
  'io/CameraExport.cpp',
  'io/CameraImport.cpp',
  'io/ExportMesh.cpp',
  'io/ImportMesh.cpp',
  'io/Load.cpp',
  'io/LoadAdd.cpp',
  'io/WriteImage.cpp',
])

ioInc = files([
  'io/BVHExport.hpp',
  'io/BVHImport.hpp',
  'io/CameraExport.hpp',
  'io/CameraImport.hpp',
  'io/ExportMesh.hpp',
  'io/ImportMesh.hpp',
  'io/Load.hpp',
  'io/LoadAdd.hpp',
  'io/WriteImage.hpp',
])


# Tracer lib
tracerSrc = files([
  'tracer/BruteForceTracer.cpp',
  'tracer/CPUTracer.cpp',
  'tracer/CUDATracer.cpp',
  'tracer/TracerBase.cpp'
])

tracerCU = files([
  'tracer/CUDAKernels.cu'
])

tracerInc = files([
  'tracer/BruteForceTracer.hpp',
  'tracer/CPUTracer.hpp',
  'tracer/CUDATracer.hpp',
  'tracer/TracerBase.hpp'
])


# Viewer lib
viewerSrc = files([
  'viewer/BVHRenderer.cpp',
  'viewer/LiveTracer.cpp',
  'viewer/MeshRenderer.cpp',
  'viewer/RendererBase.cpp',
  'viewer/Viewer.cpp',
  'viewer/Window.cpp'
])

viewerInc = files([
  'viewer/BVHRenderer.hpp',
  'viewer/LiveTracer.hpp',
  'viewer/MeshRenderer.hpp',
  'viewer/RendererBase.hpp',
  'viewer/Viewer.hpp',
  'viewer/Window.hpp'
])


# CUDA lib
cudaSrc = files([
  'cuda/CopyToGPU.cpp',
  'cuda/CopyToHost.cpp'
])

cudaCU = files([
  'cuda/cudaFN.cu'
])


#################################
#### Generate Enum2Str class ####
#################################

allInc = baseInc + builderInc + miscInc + ioInc + viewerInc
enumData = []

foreach i : allInc
#foreach i : files(['base/BVH.hpp', 'base/Command.hpp', 'base/CameraBase.hpp', 'base/State.hpp', 'viewer/RendererBase.hpp'])
  enumData += enumParser.process(i)
endforeach

enum2Str = custom_target('Enum2Str',
  input:   ['../enumGenCfg.json'] + enumData,
  output:  ['Enum2Str.hpp', 'Enum2Str.cpp'],
  command: [enumGen, '-d', '@OUTDIR@', '-p', '@SOURCE_DIR@', '-c' , '@INPUT0@', 'generate', 'Enum2Str', '@INPUT@']
)

####################
#### Build libs ####
####################

cuInclude  = []

foreach i : incDirsRaw
  cuInclude += '-I' + i
endforeach

cuInclude += '-I' + meson.build_root()

enum2StrLib = static_library('enum2Str', enum2Str,
  include_directories: incDirs,
  install: true,
  dependencies: baseDeps,
)

baseLib = static_library('base', baseSrc,
  include_directories: incDirs,
  install: true,
  link_with: [enum2StrLib, externalsLib],
  dependencies: baseDeps,
)

miscLib = static_library('misc', miscSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, enum2StrLib, externalsLib],
  dependencies: baseDeps,
)


cudaCuda = []
foreach i : cudaCU
  cudaCuda += nvccCompiler.process(i, extra_args: ['--std=c++14'] + cuInclude)
endforeach

cudaLib = static_library('cuda', cudaSrc + cudaCuda,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, miscLib, enum2StrLib, externalsLib],
  dependencies: baseDeps,
)

builderCuda = []
foreach i : builderCU
  builderCuda += nvccCompiler.process(i, extra_args: ['--std=c++14'] + cuInclude)
endforeach

builderLib = static_library('builder', builderSrc + builderCuda,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, miscLib, enum2StrLib, externalsLib],
  dependencies: baseDeps,
)

tracerCuda = []
foreach i : tracerCU
  tracerCuda += nvccCompiler.process(i, extra_args: ['--std=c++14'] + cuInclude)
endforeach

tracerLib = static_library('tracer', tracerSrc + tracerCuda,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, miscLib, enum2StrLib, externalsLib],
  dependencies: baseDeps,
)

ioLib = static_library('io', ioSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, miscLib, enum2StrLib, externalsLib],
  dependencies: baseDeps + [assimp],
)

viewerLib = static_library('viewer', viewerSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, miscLib, tracerLib, enum2StrLib, externalsLib],
  dependencies: baseDeps + [GL, glfw, dl],
)
