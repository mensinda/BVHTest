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
  'builder/BuilderBase.cpp',
  'builder/Median.cpp',
  'builder/OptimizerBase.cpp',
  'builder/Wald07.cpp'
])


builderInc = files([
  'builder/Bittner13.hpp',
  'builder/BuilderBase.hpp',
  'builder/Median.hpp',
  'builder/OptimizerBase.hpp',
  'builder/Wald07.hpp'
])


# Camera lib
miscSrc = files([
  'misc/Camera.cpp',
  'misc/Validate.cpp'
])

miscInc = files([
  'misc/Camera.hpp',
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
  'io/WriteImage.hpp',
])


# Tracer lib
tracerSrc = files([
  'tracer/BruteForceTracer.cpp',
  'tracer/CPUTracer.cpp',
  'tracer/TracerBase.cpp'
])

tracerInc = files([
  'tracer/BruteForceTracer.hpp',
  'tracer/CPUTracer.hpp',
  'tracer/TracerBase.hpp'
])


# Viewer lib
viewerSrc = files([
  'viewer/BVHRenderer.cpp',
  'viewer/MeshRenderer.cpp',
  'viewer/RendererBase.cpp',
  'viewer/Viewer.cpp',
  'viewer/Window.cpp'
])

viewerInc = files([
  'viewer/BVHRenderer.hpp',
  'viewer/MeshRenderer.hpp',
  'viewer/RendererBase.hpp',
  'viewer/Viewer.hpp',
  'viewer/Window.hpp'
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

enum2StrLib = library('enum2Str', enum2Str,
  include_directories: incDirs,
  install: true,
  dependencies: baseDeps,
)

baseLib = library('base', baseSrc,
  include_directories: incDirs,
  install: true,
  link_with: [enum2StrLib, externalsLib],
  dependencies: baseDeps,
)

builderLib = library('builder', builderSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, enum2StrLib, externalsLib],
  dependencies: baseDeps,
)

miscLib = library('misc', miscSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, enum2StrLib, externalsLib],
  dependencies: baseDeps,
)

tracerLib = library('tracer', tracerSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, miscLib, enum2StrLib, externalsLib],
  dependencies: baseDeps,
)

ioLib = library('io', ioSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, miscLib, enum2StrLib, externalsLib],
  dependencies: baseDeps + [assimp],
)

viewerLib = library('viewer', viewerSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, miscLib, tracerLib, enum2StrLib, externalsLib],
  dependencies: baseDeps + [GL, glfw, dl, glm],
)
