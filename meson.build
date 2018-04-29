project( 'BVHTest', 'cpp',
  default_options : [
    'cpp_std=c++17',
    'warning_level=3',
    'libdir=lib/BVHTest',
    'includedir=include/BVHTest',
    'b_colorout=always'
  ],
  version : '0.0.1',
  license : 'Apache 2')

externalInc = include_directories('external', 'src')
assimp      = dependency('assimp')

fullLibDirPath = get_option('prefix') + '/' + get_option('libdir')

# Build libs
subdir('src/base')

# Build exes
subdir('src/BVHTest')
subdir('test')
