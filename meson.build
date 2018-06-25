project( 'BVHTest', ['cpp', 'c'],
  default_options : [
    'cpp_std=c++17',
    'warning_level=3',
    'buildtype=debugoptimized',
    'libdir=lib/BVHTest',
    'includedir=include/BVHTest',
    'b_colorout=always'
  ],
  version : '0.0.1',
  license : 'Apache 2')

incDirsBase = ['external',  'external/fmt', 'external/cub', 'external/glm', 'external/spdlog/include', 'lib', '.', './lib']
incDirsRaw  = []

foreach i : incDirsBase
  incDirsRaw += meson.current_source_dir() + '/' + i
endforeach

incDirs = include_directories(incDirsBase)
assimp  = dependency('assimp',  required: true, version: '>=3.2.0')
glfw    = dependency('glfw3',   required: true, version: '>=3.1.0')
GL      = dependency('GL',      required: true, version: '>=3.3')
OpenMP  = dependency('openmp',  required: true)
dl      = meson.get_compiler('c').find_library('dl', required : true)
threads = dependency('threads', required: true)

if get_option('buildtype') == 'debugoptimized'
  add_global_arguments('-fno-omit-frame-pointer', language: 'cpp')
  message('Enabling perf stuff')
endif

compiler = meson.get_compiler('cpp')
cfgData  = configuration_data()
cfgData.set_quoted('BVHTEST_VERSION', meson.project_version())
cfgData.set_quoted('LOGGER_NAME',     meson.project_name())

baseDeps = [
  threads,
  OpenMP,
  compiler.find_library('stdc++fs', required: true),
  compiler.find_library('cudart', required: true, dirs: ['/usr/lib', '/usr/local/cuda/lib64', '/opt/cuda/lib64'])
]

if not compiler.has_header('filesystem') and not compiler.has_header('experimental/filesystem')
  error('Unable to find C++17 filesystem')
endif

if not ['c++17', 'c++2a'].contains(get_option('cpp_std'))
  error('At least c++17 required but ' + get_option('cpp_std') + ' provided')
endif

configure_file(
  configuration: cfgData,
  input:         'BVHTestCfg.hpp.in',
  output:        'BVHTestCfg.hpp'
)

fullLibDirPath = get_option('prefix') + '/' + get_option('libdir')

enumGen = find_program(['external/enumGen/enumGen.py', 'enumGen.py', '/usr/bin/enumGen.py', '/bin/enumGen.py'], required : true)
nvcc    = find_program(['nvcc', '/opt/cuda/bin/nvcc', '/usr/local/cuda/bin/nvcc'], required : true)
enumParser   = generator(enumGen, output: '@BASENAME@.json', arguments: ['-d', '@BUILD_DIR@', 'parse', '@INPUT@'])

nvccCompiler = generator(
  nvcc,
  output: '@BASENAME@.o',
  arguments: [
    '@EXTRA_ARGS@',
    '-gencode=arch=compute_50,code=sm_50',
    '-gencode=arch=compute_52,code=sm_52',
    '-gencode=arch=compute_60,code=sm_60',
    '-gencode=arch=compute_61,code=sm_61',
    '-gencode=arch=compute_70,code=sm_70',
    '-gencode=arch=compute_70,code=compute_70',
    '-O3', '-Xptxas', '-O3', '-lineinfo', # '--ptxas-options=-v', # '--maxrregcount=24',
    #'-O0', '-G', '-g', # '--maxrregcount=24',
    '-Xcudafe', '--diag_suppress=esa_on_defaulted_function_ignored',
    '-c', '@INPUT@', '-o', '@OUTPUT@',
])

# Build
subdir('external')
subdir('lib')
subdir('src')
#subdir('test')
