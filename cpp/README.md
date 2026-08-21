# Python vs C++ - the same LiteRT pipeline on the NPU

The same two-stage AI pipeline (person detection + per-person PPE detection,
INT8 models, inference on the Qualcomm Hexagon NPU of a Rubik Pi 3 / QCS6490)
implemented twice with a **mirrored 5-stage thread pipeline**. Both sides run on
**LiteRT** (ai-edge-litert) - Python via the `ai_edge_litert` wheel, C++ via the
LiteRT source tree - replacing the old `tensorflow`/`tensorflow-lite` runtime:

```
reader+scale ‖ model A ‖ person detect + crop/letterbox ‖ model B ‖ collector
```

| File | Implementation | Output |
|------|----------------|--------|
| `../python/benchmark.py` | Python (`ai-edge-litert`), `cv2.VideoCapture` decode | `../python/results/benchmark/python_metrics.csv`, `.../summary_python.json` |
| `main.cpp` + `run_cpp.sh` | C++17 (LiteRT built from the LiteRT source tree) | `results/cpp_<variant>_metrics.csv`, `results/summary_cpp_<variant>.json` |

The C++ binary benchmarks **two decode variants back to back in one run**:

- `software` - CPU H.264 decode (`openh264dec`). Plain `cv::VideoCapture` is
  deliberately not used: on this platform its auto-picked hardware decoder
  outputs compressed NV12 that `videoconvert` miscolors, silently breaking
  detection. Note that Python's pip `cv2` bundles FFmpeg - a faster,
  multi-threaded software decoder - so Python's wall-clock can beat this
  variant; the language overhead shows in the compute time per frame.
- `gstreamer` - an explicit GStreamer pipeline with the Qualcomm hardware
  decoder (`v4l2h264dec` + `qtivtransform`), showing the extra headroom HW
  decode buys on this platform. The colour conversion goes
  `qtivtransform ! BGRx ! videoconvert ! BGR`: `qtivtransform` is a GPU
  converter and only emits a cleanly-packed buffer in a 4-byte-aligned format
  (`BGRx`) - asking it for 24-bit `BGR` directly yields a buffer OpenCV's
  appsink mis-strides (the frame arrives replicated + interlaced and detection
  sees noise), so a trailing CPU `videoconvert` repacks `BGRx`→`BGR`.

Neither side takes CLI arguments - configuration is a block of globals at the
top of `../python/benchmark.py` and a block of constants at the top of `main.cpp`, so
runs are repeatable by design.

Both CSVs share the same columns and semantics: per-stage times are measured
inside their stage threads and overlap each other, `total_frame_ms` is the
summed compute work per frame (decode excluded - it runs in parallel), and the
headline throughput is wall-clock FPS from the summary JSON. Postprocessing
matches on both sides (person threshold 0.76, PPE threshold 0.40 + greedy
NMS). Inference times are nearly identical - it is the same NPU - the
difference is the host-side code around it.

## Repository layout

The Python baseline it is compared against lives in [../python/benchmark.py](../python/benchmark.py)
(run by the same host orchestrator). This directory is the C++ side only:

```
main.cpp         C++ implementation
CMakeLists.txt   C++ build definition
run_cpp.sh       runs the C++ binary with the same parameters as ../python/benchmark.py
../models/       INT8 .tflite models (shared, article-1 card layout)
../input/        source .mp4 video (shared)
results/         benchmark output (plain text), overwritten per run
build/           C++ build tree (gitignored)
```

## Building the C++ benchmark (on the device)

The C++ side links the **LiteRT** runtime (ai-edge-litert), built from the
LiteRT source tree via CMake (`add_subdirectory`), so you need the sources, not
a pip package. LiteRT keeps the classic TFLite C++ API unchanged - same
`tflite::Interpreter` and `TfLiteExternalDelegate*` (the QNN delegate) - only
the header prefix moved from `tensorflow/lite/...` to `tflite/...`. The CMake
target is still named `tensorflow-lite`.

LiteRT's `tflite/` CMake still needs a **TensorFlow checkout** for supporting
headers (schema / TSL / XLA). It pins **v2.21.0-rc0** (needed for the updated
`schema_generated.h`); if you don't pass `TENSORFLOW_SOURCE_DIR` it
FetchContent-downloads that tag itself. We clone it explicitly so reconfigures
don't re-download.

```bash
# 1. Toolchain + OpenCV (the apt packages are sufficient)
sudo apt update
sudo apt install -y git cmake build-essential libopencv-dev

# 2. LiteRT sources
git clone --depth 1 https://github.com/google-ai-edge/LiteRT.git ~/litert-src

# 3. TensorFlow v2.21.0-rc0 sources (supporting headers only; ~1 GB checkout)
git clone --depth 1 --branch v2.21.0-rc0 \
    https://github.com/tensorflow/tensorflow.git ~/tensorflow-2.21

# 4. Configure & build (first build downloads the remaining deps; ~1 h on device)
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release \
    -DLITERT_SOURCE_DIR=$HOME/litert-src \
    -DTENSORFLOW_SOURCE_DIR=$HOME/tensorflow-2.21
make -j$(nproc)
```

`LITERT_SOURCE_DIR` defaults to `$HOME/litert-src`, so step 4 can omit it if you
clone there. The binary lands in `build/benchmark_pipeline`. On aarch64 the
build enables `-mcpu=cortex-a78 -march=armv8.2-a+fp16+dotprod` (QCS6490 big
cores).

To use the NPU, the Hexagon delegate (`libQnnTFLiteDelegate.so` from the
Qualcomm QAIRT/QNN SDK shipped with the board) must be resolvable via
`LD_LIBRARY_PATH`. Without it both implementations fall back to CPU
(set `DELEGATE = "cpu"` at the top of `../python/benchmark.py` / `main.cpp`).

## Python setup (on the device)

```bash
cd ../python
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Run

No CLI arguments - both use video `input/1.mp4`, the hexagon delegate,
4 threads and 10 warmup runs (see the constants at the top of each file):

```bash
(cd ../python && python3 benchmark.py)
./run_cpp.sh
```

## Output (plain text, overwritten on every run)

C++ writes into `results/` here; the Python baseline writes into
`../python/results/benchmark/`.

- `cpp_software_metrics.csv`, `cpp_gstreamer_metrics.csv`, `python_metrics.csv`
  - per-frame stage timings plus CPU %, per-core %, RSS and SoC temperature
  columns
- `summary_cpp_software.json`, `summary_cpp_gstreamer.json`,
  `summary_python.json` - frames, wall-clock duration, FPS
- `env_python.json`, `telemetry_python.csv` - Python variant only (the C++
  binary reports telemetry inside its CSV)
