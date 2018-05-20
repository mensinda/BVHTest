cmdSrc = files([
  'main.cpp'
])

executable('bvhTest', cmdSrc,
  include_directories: incDirs,
  install: true,
  link_with: [enum2StrLib, externalsLib, baseLib, builderLib, ioLib, tracerLib, viewerLib, cameraLib],
  dependencies: baseDeps,
  install_rpath: fullLibDirPath
)
