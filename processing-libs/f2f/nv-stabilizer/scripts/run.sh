#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

"$ROOT_DIR/scripts/build.sh"
cp "$ROOT_DIR/build/native-release/libnv-stabilizer.so" /opt/project-system/gst-plugins/processing-libs/f2f/nv-stabilizer/lib/
cp "$ROOT_DIR"/config/* /opt/project-system/gst-plugins/processing-libs/f2f/nv-stabilizer/config/

"$ROOT_DIR/scripts/play.sh"