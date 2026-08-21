#!/bin/bash
# Stage 1 only: person detection, fully in GStreamer on the Hexagon NPU.
# No CLI args - tweak the globals below.
#
# Two passes:
#   1) MEASURE  - inference throughput with NO file output (decode -> NPU ->
#                 postprocess -> fpsdisplaysink/fakesink). Mirrors how 01-03
#                 time things: processing only, the mp4 write is NOT counted
#                 (it would be dominated by the filesystem). fpsdisplaysink's
#                 "average" starts at the first rendered frame, so the QNN
#                 init/prepare is excluded - same as load-before-loop in 01-03.
#   2) RENDER   - the annotated mp4 (boxes burned in), for visual proof. Adds
#                 qtimetamux (carry clean frame) + qtivoverlay + HW H.264 encode.

set -u
BASE="$(cd "$(dirname "$0")" && pwd)"

MODEL="$BASE/../models/foot_track_net-tflite-w8a8/foot_track_net.tflite"
LABELS="$BASE/labels/foot_track_net.txt"
SETTINGS="$BASE/labels/foot_track_net_settings.json"
INPUT="$BASE/../input/1.mp4"
RESULTS="$BASE/results"
OUTPUT="$RESULTS/stage1_boxes.mp4"
LOG="$RESULTS/stage1.log"

# LOOP: how many times pass 1 (the timed/measured pass) replays the clip.
# Default 1 keeps the quick run; the power/thermal session sets LOOP (see
# host/measure_04.py) so the trace is long enough for a steady power mean.
# Pass 2 (render) always runs once.
LOOP="${LOOP:-1}"
# RENDER=0 skips pass 2 (annotated-mp4 render) for the power session, keeping
# the trace pure inference (no HW encode / file write).
RENDER="${RENDER:-1}"

QNN_LIB="libQnnTFLiteDelegate.so"
DELEGATE_OPTS="QNNExternalDelegate,backend_type=htp;"
NOISE='getting bo cpu address|MapGbmBufInfoAddress|MESA-LOADER|failed to get driver|xcb connection|EGL display'

mkdir -p "$RESULTS"
export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}

echo "[run_stage1] input : $INPUT"
echo "[run_stage1] model : $MODEL (delegate=external / QNN HTP / NPU)"

# ---------------------------------------------------------------------------
# Pass 1: inference throughput (no file write)
# ---------------------------------------------------------------------------
echo "[run_stage1] pass 1/2: measuring inference throughput + CPU load (no encode/file), LOOP=$LOOP ..."
: > "$LOG"
python3 "$BASE/cpu_load.py" > "$RESULTS/cpu.json" & CPU_PID=$!
for _i in $(seq 1 "$LOOP"); do
gst-launch-1.0 -v -e \
  filesrc location="$INPUT" ! qtdemux ! h264parse ! v4l2h264dec ! queue ! \
  qtimlvconverter ! queue ! \
  qtimltflite model="$MODEL" delegate=external \
    external-delegate-path="$QNN_LIB" external-delegate-options="$DELEGATE_OPTS" ! queue ! \
  qtimlpostprocess module=qpd results=10 labels="$LABELS" settings="$SETTINGS" ! \
  fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
  2>&1 | grep -avE "$NOISE" >> "$LOG"
done
kill -TERM $CPU_PID 2>/dev/null; wait $CPU_PID 2>/dev/null

# fpsdisplaysink reports rolling "current" and cumulative "average" fps.
FPS=$(grep -oE 'average: ?[0-9.]+' "$LOG" | tail -1 | grep -oE '[0-9.]+')
FPS_MAX=$(grep -oE 'current: ?[0-9.]+' "$LOG" | grep -oE '[0-9.]+$' | sort -rn | head -1)
CPU=$(python3 -c "import json;print(json.load(open('$RESULTS/cpu.json'))['cpu_total_pct'])" 2>/dev/null)
PCORE=$(python3 -c "import json;print('|'.join(map(str,json.load(open('$RESULTS/cpu.json'))['per_core_pct'])))" 2>/dev/null)
RAM=$(python3 -c "import json;print(json.load(open('$RESULTS/cpu.json'))['ram_peak_mb'])" 2>/dev/null)
EOS=$(grep -c 'Got EOS' "$LOG")

# ---------------------------------------------------------------------------
# Pass 2: render annotated mp4 (visual proof; not part of the throughput number)
# ---------------------------------------------------------------------------
if [ "$RENDER" = 1 ]; then
echo "[run_stage1] pass 2/2: rendering annotated mp4 ..."
gst-launch-1.0 -e \
  filesrc location="$INPUT" ! qtdemux ! h264parse ! v4l2h264dec ! tee name=t \
  t. ! queue ! mux. \
  t. ! queue ! qtimlvconverter ! queue ! \
     qtimltflite model="$MODEL" delegate=external \
       external-delegate-path="$QNN_LIB" external-delegate-options="$DELEGATE_OPTS" ! queue ! \
     qtimlpostprocess module=qpd results=10 labels="$LABELS" settings="$SETTINGS" ! \
     text/x-raw ! queue ! mux. \
  qtimetamux name=mux ! queue ! qtivoverlay ! qtivtransform ! queue ! \
  v4l2h264enc ! h264parse ! mp4mux ! filesink location="$OUTPUT" \
  2>&1 | grep -avE "$NOISE" >> "$LOG"
else
echo "[run_stage1] pass 2/2: skipped (RENDER=0, power/thermal session)"
fi

printf '{ "stage": "stage1", "fps_inference_avg": "%s", "fps_inference_peak": "%s", "cpu_total_pct": "%s", "ram_peak_mb": "%s", "per_core_pct": "%s", "eos": %s }\n' \
  "${FPS:-NA}" "${FPS_MAX:-NA}" "${CPU:-NA}" "${RAM:-NA}" "${PCORE:-NA}" "${EOS:-0}" > "$RESULTS/stage1_summary.json"

echo "[run_stage1] done. summary:"
cat "$RESULTS/stage1_summary.json"
ls -l "$OUTPUT" 2>/dev/null || echo "[run_stage1] WARN: no annotated mp4 produced"
