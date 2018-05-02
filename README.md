# BVHTest

Source code for my Bachelor thesis

## Requirements

  - meson
  - ninja
  - assimp v4.1
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
