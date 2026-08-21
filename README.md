# Optimizing Edge AI Vision - from a single-threaded loop to a full-HW pipeline

Companion code for the second article. Article 1 asked *where* to run inference
(CPU vs NPU) and settled on the Rubik Pi 3's Hexagon NPU. This article keeps the
same two-stage AI pipeline - person detection → crop → PPE (helmet/vest)
detection, INT8 TFLite models - fixed on that NPU and asks a different question:

> once the NPU is a given, how much throughput can software architecture buy -
> and what is the *right* pipeline to leave running, not just the fastest one?

Everything runs on the **Rubik Pi 3 (NPU)**. Each stage takes the previous one's
best idea further, from a naive Python loop to a pipeline that lives entirely
inside GStreamer on dedicated hardware blocks.

## The optimization path

The code is grouped **by implementation language** - [python/](python/), [cpp/](cpp/),
[gstreamer/](gstreamer/) - over a single shared [input/](input/) (video clips) and
[models/](models/) (the INT8 `.tflite` + `labels.txt` + `metadata.json`, same card
layout as article 1). The article's optimization path runs *across* those dirs:

| Stage | Where | What it adds | Result |
|-------|-------|--------------|--------|
| Single-threaded baseline | [python/run_single.py](python/run_single.py) | A naive frame-by-frame loop - the NPU sits idle between frames | Baseline FPS |
| 5-stage thread pipeline | [python/run_pipelined.py](python/run_pipelined.py) | reader ‖ model 1 ‖ crop ‖ model 2 ‖ collector, overlapping CPU work with NPU inference | Wall-clock throughput **more than doubles** - pipelining keeps the NPU fed |
| C++ rewrite | [cpp/](cpp/) (+ [python/benchmark.py](python/benchmark.py) as the Python baseline) | The same 5-stage pipeline in C++ (LiteRT), software vs GStreamer HW video decode | With HW decode, C++ clears the Python pipeline at markedly lower CPU and RAM - the raw-speed ceiling |
| Full-HW GStreamer | [gstreamer/](gstreamer/) | Hand off *everything* to a Qualcomm `qti-ml` graph: HW H.264 decode → NPU (both models) → on-GPU crop & overlay → HW encode. No per-frame CPU code | Slower in FPS, but less than half the C++'s CPU load and the lowest memory of the three |
| Endurance | [long-term/](long-term/) | Hold each variant under a **sustained load** and record temperature, SoC clock (throttling) and wall power over time | The endurance verdict - see below |

The two Python pipeline drivers (`run_pipelined.py`, `benchmark.py`) share their
inference stages via [python/pipeline.py](python/pipeline.py) and all detection/
telemetry code via [python/common.py](python/common.py) - no duplicated pipeline.

Every variant writes its own throughput, CPU/RAM and thermal metrics to
`results/` (schema in each directory's README). The consolidated, measured
figures - together with the power and energy-per-frame numbers from the external
meter - are reported in the article itself.

## The endurance question (`long-term/`)

Peak FPS is a sprint number. The [long-term/](long-term/) test is the marathon:
each variant is held under load for a sustained run while a background sampler
logs SoC temperature, per-core clock and CPU%, with wall power captured on a
ChargerLAB (Power-Z) meter. The thesis the article lands on:

- **C++ is the fastest** but under a long run it heats up, the SoC starts to throttle and power stays high.
- **The full-HW GStreamer pipeline offloads almost everything to the NPU/GPU**,
  so it holds a flat, cool, steady line - it gives up peak FPS but is the one you
  would actually leave running 24/7, and it leaves CPU and memory free for other
  work.

So the article's answer isn't "use whichever is fastest": it's that the *best*
pipeline depends on whether you are benchmarking or deploying.

## Hardware & models

- **Device:** Rubik Pi 3, inference on the Hexagon NPU (QNN external delegate,
  `libQnnTFLiteDelegate.so`, HTP backend). The C++ build links LiteRT
  (ai-edge-litert) built from source on the device (see [cpp/README.md](cpp/README.md)).
- **Models:** the same open-license INT8 (w8a8) models as article 1 -
  `foot_track_net` (person/foot detection) and `gear_guard_net` (PPE detection).
- **Power:** measured externally with a ChargerLAB Power-Z meter (never in
  software); the run windows are sliced out of the recording by timestamp.

## Conventions

- Benchmarks run **without arguments** - parameters are global variables at the
  top of each file, for reproducibility.
- Every device run writes text metrics (CSV + `summary.json` + `env.json`) to
  `results/`; generated artifacts go to `results/`, `output/`, `graphs/` - never
  loose in the repo.
- Video clips and models are **shared** across all variants - `input/` and
  `models/` at the repo root; each language dir references them via `../`.
