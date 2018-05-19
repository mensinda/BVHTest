extLibSrc = files([
  'gl3w.c',
  'lodepng/lodepng.cpp'
])

externalsLib = library('externalsLib', extLibSrc,
  include_directories: incDirs,
  install: true,
  dependencies: baseDeps + [GL, dl]
)
