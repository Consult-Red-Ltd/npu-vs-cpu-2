# 04 - GStreamer full-HW pipeline (Rubik Pi 3, NPU)

How much can we hand to GStreamer? This experiment runs the full **two-stage
PPE cascade** (person → crop → helmet/vest) as a Qualcomm `qti-ml` GStreamer
graph: hardware H.264 decode, both inferences on the Hexagon NPU via
`qtimltflite`, ROI cropping between stages, on-GPU box overlay, hardware H.264
encode to mp4. Nothing leaves the GStreamer pipeline - no Python, no per-frame
CPU work in our code.

Same open-license models as 01–03 (`foot_track_net`, `gear_guard_net`).
Stage-1 decode uses Qualcomm's stock `qpd` module (it is the native foot_track
decoder). Stage-2 needs a **custom postprocessing module** (see below).

## Scripts

| Script | What |
|--------|------|
| `run_cascade.sh` | person + PPE cascade → mp4 |
| `run_stage1.sh` | person detection only → mp4 |

The second stage is almost free: `roi-batch-cumulative` batches all person
crops into one `gear_guard` invocation, and the NPU + pipelining absorb it, so
the full cascade runs at barely lower throughput than stage 1 alone.

## Pipeline (cascade)

```
v4l2h264dec ─┬─ queue ───────────────────────────────┐ (clean frame)
             │                                        ▼
             └─ qtimlvconverter → qtimltflite(foot_track,NPU)
                → qtimlpostprocess(qpd) ──────────► qtimetamux1 ─┬─ queue ──────────────┐ (frame + person ROIs)
                                                                 │                      ▼
                                                                 └─ qtimlvconverter      qtimetamux2 → qtivoverlay
                                                                    mode=roi-batch-cumulative    ▲       → v4l2h264enc
                                                                    → qtimltflite(gear_guard,NPU)│       → mp4mux → filesink
                                                                    → qtimlpostprocess(gearguard)┘
```

`qtimetamux` carries the clean frame next to the detection metadata; the
stage-2 converter reads the stage-1 person ROIs and crops them for
`gear_guard`; `qtivoverlay` draws both person and PPE boxes; the rest is the HW
encoder.

## The custom `gearguard` postprocessing module

`gear_guard_net-ppe-detection-w8a8` emits three already-decoded, pre-NMS tensors
(`boxes[1,3780,4]`, `scores[1,3780]`, `class_idx[1,3780]`). No stock `qti`
detection module consumes that layout (yolov8/yolov5 want one `[1,4+C,N]`
tensor; yolo-nas wants per-class scores; ssd-mobilenet wants a count tensor -
Qualcomm's own PPE sample assumes a different, raw export). So `module/`
contains a small custom module - threshold + per-class NMS on the decoded
outputs, the same logic as 01–03 - built and installed on the device:

```bash
cd module && IMSDK=~/gst-plugins-imsdk bash build_and_install.sh
```

`qtimlpostprocess` loads modules by `dlopen` from
`/usr/lib/aarch64-linux-gnu/imsdk/qtimlpostprocess/modules/`, so this is a
single ~150-line `.so` (links only libstdc++) - no SDK or plugin rebuild, no
cross-toolchain. Build it **natively on the Rubik** (`g++ -shared`).

## Config files (authored, not shipped by Qualcomm)

Qualcomm does not publish the label/settings JSONs for these models (the model
`.tflite` comes from AI Hub; the configs are demo-specific). They live in
`labels/`, reverse-engineered from the module sources:

- `foot_track_net.txt` / `gear_guard_net.txt` - class labels, one per line.
- `foot_track_net_settings.json` - `qpd` settings: `confidence` threshold (%) and
  the 17 landmark names per class. **Every key must exist** - the qti JSON parser
  uses `map::at`, so a missing `id`/`name`/`landmarks` key crashes the element.
- `gear_guard_net_settings.json` - `confidence` threshold (%) for the PPE stage.

## Run

```bash
cd module && bash build_and_install.sh && cd ..
bash run_cascade.sh       # results/cascade_boxes.mp4 + cascade_summary.json
bash run_stage1.sh        # results/stage1_boxes.mp4  + stage1_summary.json
```

## Output

`results/`:

- `cascade_boxes.mp4` - input video with person + helmet/vest boxes burned in
- `cascade_summary.json` - inference throughput (`fps_inference_avg` / `_peak`)
- `cascade.log` - filtered GStreamer log
- (`stage1_*` equivalents when running stage 1 only)

**Throughput methodology** (matches 01–03): FPS is measured on a separate
pass that ends in `fakesink` - decode → NPU → postprocess, **no overlay, no
H.264 encode, no file write** - so the filesystem never skews the number.
`fpsdisplaysink`'s average starts at the first rendered frame, so the QNN
prepare/init is excluded (the same way 01–03 load the model before the timed
loop). The mp4 with boxes is produced by a second, untimed render pass.

## How the full-HW cascade compares to the C++/Python pipelines

Same two models on the NPU, HW decode, file write excluded from timing. The
measured throughput, CPU and RAM figures for all variants live in the article;
the mechanism behind them is what matters here.

RAM is reported as peak resident set (RSS). The DMA/ION and NPU-context buffers
live outside RSS, so the true footprint is higher for every NPU path - but RSS
is a consistent process-memory proxy across all three.

**Why the cascade is slower in FPS.** Both pipelines overlap stages across
frames, so wall time ≈ the slowest point on the critical path. The hand-tuned
C++ 5-thread pipeline overlaps model A *behind* model B (its slowest stage), so
A is effectively free and wall ≈ inference B. The GStreamer cascade instead runs
the two NPU inferences back-to-back (wall ≈ inference A + B), and the stock `qti`
elements add per-buffer cost the lean C++ avoids: two GPU `qtimlvconverter`
conversions, two `qtimetamux` metadata muxes, detections serialized to
`text/x-raw`, and the `roi-batch-cumulative` crop coupling stage 2 to stage 1.

**But the cascade uses far less CPU and the least RAM.** It pushes image→tensor
conversion to the GPU and both inferences to the NPU, so the CPU mostly
orchestrates. It is also the leanest in memory: the gst-launch pipeline never
links OpenCV, while the C++ binary does (and its HW-decode path pulls in the
GPU/EGL Adreno stack on top); Python carries the interpreter + LiteRT + numpy +
cv2. So the trade is real and many-sided: the hand-tuned C++ is the raw-speed
ceiling; the cascade is the **lowest-CPU, lowest-RAM, most-offloaded** option
with almost no custom code (one ~150-line postproc module) and in-pipeline
overlay + HW encode to mp4 - leaving CPU and memory free for other work.

## Notes

- NPU is selected with `delegate=external` + `libQnnTFLiteDelegate.so` +
  `QNNExternalDelegate,backend_type=htp` (this plugin build has no `hexagon`
  delegate nick; the QNN external delegate is the NPU path).
- Headless (no Weston) `qtivoverlay` spams harmless `GBM mmap` warnings on
  stderr - filtered out of the saved log; the overlay still composes correctly.
- **`qtivtransform` before the encoder** is required: `v4l2h264dec`/`v4l2h264enc`
  leave a garbage (green) block in a corner of the frame (a buffer
  stride/alignment artifact, reproducible with a bare decode→encode pipeline).
  A hardware `qtivtransform` between `qtivoverlay` and `v4l2h264enc` rewrites a
  clean, aligned buffer and removes it.
