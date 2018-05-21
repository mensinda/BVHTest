cmdSrc = files([
  'main.cpp'
])

executable('bvhTest', cmdSrc,
  include_directories: incDirs,
  install: true,
  link_with: [enum2StrLib, externalsLib, baseLib, builderLib, ioLib, tracerLib, viewerLib, miscLib],
  dependencies: baseDeps,
  install_rpath: fullLibDirPath
)
