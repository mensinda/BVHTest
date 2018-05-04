#################
#### Sources ####
#################

# Base lib
baseSrc = files([
  'base/CameraBase.cpp',
  'base/Configurable.cpp',
  'base/Config.cpp',
  'base/Command.cpp',
  'base/State.cpp',
  'base/StatusDump.cpp'
])

baseInc = files([
  'base/CameraBase.hpp',
  'base/Configurable.hpp',
  'base/Config.hpp',
  'base/Command.hpp',
  'base/State.hpp',
  'base/StatusDump.hpp',
  'base/StringHash.hpp'
])


# Base lib
cameraSrc = files([
  'camera/Camera.cpp'
])

cameraInc = files([
  'camera/Camera.hpp'
])


# IO lib
ioSrc = files([
  'io/ExportMesh.cpp',
  'io/ImportMesh.cpp',
  'io/Load.cpp'
])

ioInc = files([
  'io/ExportMesh.hpp',
  'io/ImportMesh.hpp',
  'io/Load.hpp'
])


# Viewer lib
viewerSrc = files([
  'viewer/gl3w.c',
  'viewer/MeshRenderer.cpp',
  'viewer/Viewer.cpp',
  'viewer/Window.cpp'
])

viewerInc = files([
  'viewer/MeshRenderer.hpp',
  'viewer/Viewer.hpp',
  'viewer/Window.hpp'
])


#################################
#### Generate Enum2Str class ####
#################################

allInc = baseInc + ioInc + viewerInc
enumData = []

foreach i : allInc
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
  link_with: [enum2StrLib],
  dependencies: baseDeps,
)

cameraLib = library('camera', cameraSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, enum2StrLib],
  dependencies: baseDeps,
)

ioLib = library('io', ioSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, enum2StrLib],
  dependencies: baseDeps + [assimp],
)

viewerLib = library('viewer', viewerSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, cameraLib, enum2StrLib],
  dependencies: baseDeps + [GL, glfw, dl, glm],
)
