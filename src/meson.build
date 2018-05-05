cmdSrc = files([
  'main.cpp'
])

executable('bvhTest', cmdSrc,
  include_directories: incDirs,
  install: true,
  link_with: [enum2StrLib, baseLib, builderLib, ioLib, viewerLib, cameraLib],
  dependencies: [threads],
  install_rpath: fullLibDirPath
)
