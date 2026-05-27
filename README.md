# gst-plugins

GStreamer plugins for frame-to-frame and frame-to-data processing.
Each plugin loads a shared library that implements the actual processing.
Each plugin and each processing library can be built and distributed separately.

## Build

```bash
cmake --preset native-debug
cmake --build --preset native-debug
cmake --build --preset native-debug --target package
# Or via script:
./scripts/build.sh native debug
```

Available presets:

- cross-debug
- cross-release
