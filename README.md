# BVHTest

Source code for my Bachelor thesis

## Requirements

  - meson
  - ninja
  - assimp >= 4.1
  - OpenGL >= 3.3
  - GLFW   >= 3.1
  - CUDA   >= 9.1
  - a C++17 compiler

## Building

```bash
cd $PROJECT_ROOT
meson build         # only once
ninja -C build
```

# Usage

Run with `./build/src/bvhTest`

BVHTest has only a minimal cmd. It is designed to execute JSON configuration files.

Use `./build/src/bvhTest list` to list available commands, then generate a sample
config file with `./build/src/bvhTest genrate <OUT FILE> [LIST OF COMMANDS]`.

Use `./build/src/bvhTest run <FILE>` to execute the commands for each input in baseConfig.input

# Converting binary files

## Mesh files

Binary mesh files (`data.json`, `data.bin`) can be converted to the new file format version with:

```json
{
  "baseConfig": {
    "basePath": "/path/to/meshes",
    "input": [
      "list",
      "of",
      "files",
      "to",
      "convert"
    ],
    "maxThreads": 4,
    "name": "test",
    "verbose": true
  },
  "commands": [
    {
      "cmd": "load"
    },
    {
      "cmd": "export",
      "options": {
        "outDir": "/path/to/meshes"
      }
    },
    {
      "cmd": "status"
    }
  ]
}
```

## Saved BVH files

These files can not be upgraded to a new version.
