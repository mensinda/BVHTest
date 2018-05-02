project( 'BVHTest', 'cpp',
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

incDirs = include_directories('external', 'external/spdlog/include', 'lib', '.', './lib')
assimp  = dependency('assimp')
threads = dependency('threads')

compiler = meson.get_compiler('cpp')
cfgData  = configuration_data()
cfgData.set_quoted('BVHTEST_VERSION', meson.project_version())
cfgData.set_quoted('LOGGER_NAME',     meson.project_name())

baseDeps = [threads]

if compiler.has_header('filesystem')
  cfgData.set('FS_INC', 'filesystem')
  cfgData.set('FS_NS',  'std::filesystem')
  message('Using C++17 filesystem')
elif compiler.has_header('experimental/filesystem')
  cfgData.set('FS_INC', 'experimental/filesystem')
  cfgData.set('FS_NS',  'std::experimental::filesystem')
  baseDeps += compiler.find_library('stdc++fs', required: true)
  message('Using C++17 EXPERIMENTAL filesystem')
else
  error('Unable to find C++17 filesystem')
endif

configure_file(
  configuration: cfgData,
  input:         'BVHTestCfg.hpp.in',
  output:        'BVHTestCfg.hpp'
)

fullLibDirPath = get_option('prefix') + '/' + get_option('libdir')

enumGen = find_program(['external/enumGen/enumGen.py', 'enumGen.py', '/usr/bin/enumGen.py', '/bin/enumGen.py'], required : true)
enumParser = generator(enumGen, output: '@BASENAME@.json', arguments: ['-d', '@BUILD_DIR@', 'parse', '@INPUT@'])

# Build
subdir('lib')
subdir('src')
subdir('test')
