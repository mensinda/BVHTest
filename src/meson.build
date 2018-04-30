cmdSrc = files([
  'main.cpp'
])

executable('bvhTest', cmdSrc,
  include_directories: incDirs,
  install: true,
  link_with: [enum2StrLib, baseLib, ioLib],
  dependencies: [threads],
  install_rpath: fullLibDirPath
)
