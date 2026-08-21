#!/bin/bash
# Full two-stage PPE cascade, entirely in GStreamer on the Hexagon NPU.
# No CLI args - tweak the globals below.
#
#   stage 1: foot_track_net (NPU) -> qpd        -> person ROIs
#   crop   : qtimlvconverter mode=roi-batch-cumulative  (crops each person)
#   stage 2: gear_guard_net (NPU) -> gearguard  -> helmet/vest on each crop
#   draw   : qtivoverlay        encode: v4l2h264enc -> mp4
#
# Two passes (same as 01-03 methodology): MEASURE ends in fakesink (no encode,
# no file write, QNN init excluded via fpsdisplaysink); RENDER writes the mp4.

set -u
BASE="$(cd "$(dirname "$0")" && pwd)"

FOOT="$BASE/../models/foot_track_net-tflite-w8a8/foot_track_net.tflite"
GEAR="$BASE/../models/gear_guard_net-tflite-w8a8/gear_guard_net.tflite"
L="$BASE/labels"
# INPUT is overridable so the long-term session can point a single, continuous
# gst-launch at one long clip in /dev/shm (gst-launch can't loop a demuxed file
# in-process, so a long file is how we get one uninterrupted run - no per-LOOP
# pipeline restart, no reload sawtooth). Default stays the short repo clip.
INPUT="${INPUT:-$BASE/../input/1.mp4}"
RESULTS="$BASE/results"
OUTPUT="$RESULTS/cascade_boxes.mp4"
LOG="$RESULTS/cascade.log"

# LOOP: how many times pass 1 (the timed/measured pass) replays the clip.
# Default 1 keeps the quick run; the power/thermal session sets LOOP (see
# host/measure_04.py) so the trace is long enough for a steady power mean and
# a realistic SoC temperature. Pass 2 (render) always runs once.
# RUN_SECONDS>0 caps the (single) measured pass at that many seconds. gst-launch
# can't self-time, so we wrap it in `timeout --signal=INT`; combined with the
# `-e` flag below, the interrupt forces a clean EOS so fpsdisplaysink still
# prints the average FPS. The long-term session sets this (with LOOP=1 + a long
# /dev/shm clip) for one continuous, dip-free run.
RUN_SECONDS="${RUN_SECONDS:-0}"
GST_TIMEOUT=""
[ "$RUN_SECONDS" != "0" ] && GST_TIMEOUT="timeout --signal=INT $RUN_SECONDS"

LOOP="${LOOP:-1}"
# RENDER=0 skips pass 2 (the annotated-mp4 render). The power session sets
# RENDER=0 so the trace stays pure inference - HW encode + file write would
# skew it, just like 01-03 exclude the file write from their timing.
RENDER="${RENDER:-1}"

QNN_LIB="libQnnTFLiteDelegate.so"
OPTS="QNNExternalDelegate,backend_type=htp;"
NOISE='getting bo cpu address|MapGbmBufInfoAddress|MESA-LOADER|failed to get driver|xcb connection|EGL display|GEM Handle|tiling.h|ConvLayer|AvgPool|Concat|crouton|op_registry|tcm_migration|DDR bandwidth|spill_|fill_|write_total|read_total|Completed stage|Starting stage| <W>| <E>|reduce_opts|concat_opts'

mkdir -p "$RESULTS"
export LD_LIBRARY_PATH=/usr/lib:${LD_LIBRARY_PATH:-}

echo "[cascade] stage1=foot_track(qpd)  stage2=gear_guard(gearguard)  both NPU"

# ---------------------------------------------------------------------------
# Pass 1: cascade inference throughput (no encode / no file)
# ---------------------------------------------------------------------------
echo "[cascade] pass 1/2: measuring throughput + CPU load (no encode/file), LOOP=$LOOP ..."
: > "$LOG"
python3 "$BASE/cpu_load.py" > "$RESULTS/cpu.json" & CPU_PID=$!
for _i in $(seq 1 "$LOOP"); do
$GST_TIMEOUT gst-launch-1.0 -v -e \
  filesrc location="$INPUT" ! qtdemux ! h264parse ! v4l2h264dec ! tee name=t0 \
  t0. ! queue ! mux1. \
  t0. ! queue ! qtimlvconverter ! queue ! \
     qtimltflite model="$FOOT" delegate=external external-delegate-path="$QNN_LIB" external-delegate-options="$OPTS" ! queue ! \
     qtimlpostprocess module=qpd results=10 labels="$L/foot_track_net.txt" settings="$L/foot_track_net_settings.json" ! text/x-raw ! queue ! mux1. \
  qtimetamux name=mux1 ! queue ! tee name=t1 \
  t1. ! queue ! mux2. \
  t1. ! queue ! qtimlvconverter mode=roi-batch-cumulative ! queue ! \
     qtimltflite model="$GEAR" delegate=external external-delegate-path="$QNN_LIB" external-delegate-options="$OPTS" ! queue ! \
     qtimlpostprocess module=gearguard results=10 labels="$L/gear_guard_net.txt" settings="$L/gear_guard_net_settings.json" ! text/x-raw ! queue ! mux2. \
  qtimetamux name=mux2 ! queue ! fpsdisplaysink video-sink=fakesink text-overlay=false sync=false \
  2>&1 | grep -avE "$NOISE" >> "$LOG"
done
kill -TERM $CPU_PID 2>/dev/null; wait $CPU_PID 2>/dev/null

# Throughput, frame-weighted across ALL replays. fpsdisplaysink reports a
# CUMULATIVE average per gst-launch process, and LOOP=N runs N processes into
# this one log, each restarting its own counters. Reading `average` from the last
# line therefore described the LAST replay alone - with LOOP=12 over the
# 223-frame clip that is ~5 samples over ~2.5 s, which is both noisy (stdev 4.9
# FPS over 10 identical repeats) and biased low, since every replay pays
# pipeline setup inside its own measurement. Summing frames and per-replay
# elapsed (frames / that replay's average) gives the rate over everything
# actually measured. A replay boundary is where `rendered` drops.
AGG=$(awk '
  match($0, /rendered: ?[0-9]+, dropped: ?[0-9]+, current: ?[0-9.]+, average: ?[0-9.]+/) {
    split(substr($0, RSTART, RLENGTH), f, /[:,]/)
    r = f[2] + 0; a = f[8] + 0
    if (r < prev_r) { frames += prev_r; if (prev_a > 0) secs += prev_r / prev_a; n++ }
    prev_r = r; prev_a = a
  }
  END {
    if (prev_r > 0) { frames += prev_r; if (prev_a > 0) secs += prev_r / prev_a; n++ }
    printf "%.2f %d %d", (secs > 0 ? frames / secs : 0), frames, n
  }' "$LOG")
FPS=$(echo "$AGG" | cut -d' ' -f1)
FRAMES=$(echo "$AGG" | cut -d' ' -f2)
REPLAYS=$(echo "$AGG" | cut -d' ' -f3)
FPS_MAX=$(grep -oE 'current: ?[0-9.]+' "$LOG" | grep -oE '[0-9.]+$' | sort -rn | head -1)
CPU=$(python3 -c "import json;print(json.load(open('$RESULTS/cpu.json'))['cpu_total_pct'])" 2>/dev/null)
PCORE=$(python3 -c "import json;print('|'.join(map(str,json.load(open('$RESULTS/cpu.json'))['per_core_pct'])))" 2>/dev/null)
RAM=$(python3 -c "import json;print(json.load(open('$RESULTS/cpu.json'))['ram_peak_mb'])" 2>/dev/null)

# ---------------------------------------------------------------------------
# Pass 2: render annotated mp4 (person + PPE boxes burned in)
# ---------------------------------------------------------------------------
if [ "$RENDER" = 1 ]; then
echo "[cascade] pass 2/2: rendering annotated mp4 ..."
gst-launch-1.0 -e \
  filesrc location="$INPUT" ! qtdemux ! h264parse ! v4l2h264dec ! tee name=t0 \
  t0. ! queue ! mux1. \
  t0. ! queue ! qtimlvconverter ! queue ! \
     qtimltflite model="$FOOT" delegate=external external-delegate-path="$QNN_LIB" external-delegate-options="$OPTS" ! queue ! \
     qtimlpostprocess module=qpd results=10 labels="$L/foot_track_net.txt" settings="$L/foot_track_net_settings.json" ! text/x-raw ! queue ! mux1. \
  qtimetamux name=mux1 ! queue ! tee name=t1 \
  t1. ! queue ! mux2. \
  t1. ! queue ! qtimlvconverter mode=roi-batch-cumulative ! queue ! \
     qtimltflite model="$GEAR" delegate=external external-delegate-path="$QNN_LIB" external-delegate-options="$OPTS" ! queue ! \
     qtimlpostprocess module=gearguard results=10 labels="$L/gear_guard_net.txt" settings="$L/gear_guard_net_settings.json" ! text/x-raw ! queue ! mux2. \
  qtimetamux name=mux2 ! queue ! qtivoverlay ! qtivtransform ! queue ! \
  v4l2h264enc ! h264parse ! mp4mux ! filesink location="$OUTPUT" \
  2>&1 | grep -avE "$NOISE" >> "$LOG"
else
echo "[cascade] pass 2/2: skipped (RENDER=0, power/thermal session)"
fi

EOS=$(grep -c 'Got EOS' "$LOG")
printf '{ "stage": "cascade", "fps_inference_avg": "%s", "fps_inference_peak": "%s", "frames": %s, "replays": %s, "cpu_total_pct": "%s", "ram_peak_mb": "%s", "per_core_pct": "%s", "eos": %s }\n' \
  "${FPS:-NA}" "${FPS_MAX:-NA}" "${FRAMES:-0}" "${REPLAYS:-0}" "${CPU:-NA}" "${RAM:-NA}" \
  "${PCORE:-NA}" "${EOS:-0}" > "$RESULTS/cascade_summary.json"

echo "[cascade] done. summary:"
cat "$RESULTS/cascade_summary.json"
ls -l "$OUTPUT" 2>/dev/null || echo "[cascade] WARN: no annotated mp4 produced"
