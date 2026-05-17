# gst-plugins

This workspace now supports a root CMake superbuild that always builds all discovered projects together.

## Build From Root With Presets

Configure and build all projects:

```bash
cmake --preset native-debug
cmake --build --preset native-debug
cmake --build --preset native-debug --target package
```

Cross presets are available:

- cross-debug
- cross-release

## Build From Root Script

The root wrapper always builds and packages all discovered projects:

```bash
./scripts/build.sh <native|cross> <debug|release>
```

Examples:

```bash
./scripts/build.sh native debug
```

## Add a New Subproject

1. Add a subdirectory that contains both CMakeLists.txt and CMakePresets.json.
2. Reconfigure root preset so the superbuild re-discovers subprojects.
3. Build using the default build target and package from root.
