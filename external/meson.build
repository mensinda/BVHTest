extLibSrc = files([
  'gl3w.c',
  'lodepng/lodepng.cpp',
  'minilzo-2.10/minilzo.c'
])

externalsLib = library('externalsLib', extLibSrc,
  include_directories: incDirs,
  install: true,
  dependencies: baseDeps + [GL, dl]
)
