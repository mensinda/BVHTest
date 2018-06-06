extLibSrc = files([
  'gl3w.c',
  'lodepng/lodepng.cpp',
  'minilzo-2.10/minilzo.c',
  'fmt/fmt/format.cc',
  'fmt/fmt/ostream.cc',
  'fmt/fmt/posix.cc',
  'fmt/fmt/printf.cc'
#  'fmt/src/format.cc',
#  'fmt/src/posix.cc'
])

externalsLib = static_library('externalsLib', extLibSrc,
  include_directories: incDirs,
  install: true,
  dependencies: baseDeps + [GL, dl]
)
