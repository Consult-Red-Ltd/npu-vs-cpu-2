#!/bin/bash
# Build the custom gear_guard postprocessing module natively on the Rubik and
# install it into qtimlpostprocess's dlopen module directory. Runs ON the device.
#
# It is a single standalone .so (links only libstdc++) loaded by qtimlpostprocess
# when module=gearguard is requested. No SDK/plugin rebuild, no cross-toolchain.
set -e

BASE="$(cd "$(dirname "$0")" && pwd)"
IMSDK="${IMSDK:-$HOME/gst-plugins-imsdk}"
INC="$IMSDK/gst-plugin-mlpostprocess/modules"
MODULES_DIR="/usr/lib/aarch64-linux-gnu/imsdk/qtimlpostprocess/modules"
SO="libml-postprocess-gearguard.so"

if [ ! -d "$INC" ]; then
  echo "ERROR: IM SDK headers not found at $INC (set IMSDK=/path/to/gst-plugins-imsdk)"
  exit 1
fi

echo "[build] g++ -shared $SO (headers from $INC)"
g++ -shared -fPIC -std=c++17 -O2 \
  "$BASE/ml-postprocess-gearguard.cc" \
  -I"$INC" -I"$INC/object-detection" \
  -o "$BASE/$SO"

echo "[install] sudo cp $SO -> $MODULES_DIR"
sudo cp "$BASE/$SO" "$MODULES_DIR/"
sudo chmod 0755 "$MODULES_DIR/$SO"
ls -l "$MODULES_DIR/$SO"
echo "[done] module 'gearguard' available to qtimlpostprocess"
