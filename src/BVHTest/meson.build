cmdSrc = [
  'main.cpp',
  'Config.cpp'
]

cmdInc = [
  'Config.hpp'
]

executable('bvhTest', cmdSrc,
  include_directories: externalInc,
  install: true,
  link_with: [baseLib],
  install_rpath: fullLibDirPath
)
