#!/bin/bash
set -e

# Environment passed through to the binary (inherited by the child process):
#   LOOP_RUNS=N        replay the clip N times (long run for power/thermal)
#   DECODE_VARIANT=x   run only "software" or "gstreamer" (default: both)
# e.g.  DECODE_VARIANT=gstreamer LOOP_RUNS=30 ./run_cpp.sh

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR"
./build/benchmark_pipeline
