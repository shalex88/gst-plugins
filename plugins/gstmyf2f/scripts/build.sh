#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
TYPE=${1:-native}

if [ "$TYPE" == "cross" ]; then
  . /mnt/bsp/bsp/sdk-orin-scarthgap/environment-setup-armv8a-oe4t-linux
fi

cd ${ROOT_DIR}
cmake --preset "${TYPE}"-release
cmake --build --preset "${TYPE}"-release