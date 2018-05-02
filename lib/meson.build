#################
#### Sources ####
#################

# Base lib
baseSrc = files([
  'base/Configurable.cpp',
  'base/Config.cpp',
  'base/Command.cpp',
  'base/State.cpp'
])

baseInc = files([
  'base/Configurable.hpp',
  'base/Config.hpp',
  'base/Command.hpp',
  'base/State.hpp',
  'base/StringHash.hpp'
])


# IO lib
ioSrc = files([
  'io/ExportMesh.cpp',
  'io/ImportMesh.cpp'
])

ioInc = files([
  'io/ExportMesh.hpp',
  'io/ImportMesh.hpp'
])


#################################
#### Generate Enum2Str class ####
#################################

allInc = baseInc + ioInc
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

ioLib = library('io', ioSrc,
  include_directories: incDirs,
  install: true,
  link_with: [baseLib, enum2StrLib],
  dependencies: baseDeps + [assimp],
)
