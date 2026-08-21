# Single-threaded vs 5-stage pipeline - LiteRT on the NPU

The same two-stage AI pipeline (person detection + per-person PPE detection,
INT8 models on **LiteRT** / ai-edge-litert, inference on the Qualcomm Hexagon
NPU) executed in two ways on the same device (Rubik Pi 3, QCS6490):

| Script | Variant |
|--------|---------|
| `run_single.py` | everything sequential in one thread: decode → model 1 → crop → model 2 → write |
| `run_pipelined.py` | 5-stage thread pipeline: reader+scale ‖ model 1 ‖ crop+prep ‖ model 2 ‖ writer |

The story this experiment tells: run everything in one thread first, look at
the per-stage breakdown of a frame (`frames.csv` records every stage), notice
that the NPU sits idle while the CPU decodes, crops and writes - then
restructure into a pipeline where every stage works on a different frame at
the same time: while frame *n* sits in model 2, frame *n+1* is being cropped,
frame *n+2* is in model 1 and frame *n+3* is being decoded. The per-frame
work is identical - the NPU just never waits for the CPU stages around it.

Compared metrics: wall-clock FPS, per-frame stage timings, CPU utilisation
(total and per core), SoC temperature, RSS. Power draw can be measured
externally with a USB power meter - both variants run on the same device, so
no cable juggling is needed: one recording per variant.

## Repository layout

```
common.py         shared helpers (LiteRT I/O, postprocessing, telemetry, env dump)
pipeline.py       shared 5-stage inference stages (used by run_pipelined + benchmark)
run_single.py     sequential variant
run_pipelined.py  5-stage pipeline variant
benchmark.py      pipelined variant used as the Python baseline for the C++ comparison (see ../cpp/)
../models/        INT8 .tflite models (shared, article-1 card layout)
../input/         source .mp4 videos (shared)
results/          benchmark output (plain text), results/<variant>/, overwritten per run
output/           annotated videos (only when SAVE_VIDEO = True; gitignored)
```

## Setup (on the device)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The Hexagon delegate (`libQnnTFLiteDelegate.so`, Qualcomm QAIRT/QNN SDK) must
be resolvable via `LD_LIBRARY_PATH`; otherwise the scripts fall back to CPU.

## Run

No CLI arguments - parameters are globals at the top of each script:

```bash
python run_single.py
python run_pipelined.py
```

`SAVE_VIDEO = True` writes annotated videos to `output/<variant>/`.

`TRACE=1 python run_pipelined.py` additionally records, per frame, the
wall-clock `[start,end]` each stage spent on it (buffered in memory, written once
to `results/pipelined/trace.csv` at the end). Off by default; pure timestamp 
bookkeeping, no effect on the measured stage times.

## Output (plain text, overwritten on every run)

Each run writes to `results/single/` or `results/pipelined/`:

- `frames.csv` - per frame: stage timings, person count, SoC temperature
  (same columns as in the CPU-vs-NPU experiment)
- `telemetry.csv` - every 200 ms: `elapsed_s`, `cpu_percent`,
  `per_core_percent` (pipe-separated), `rss_mb`, `temp_c`
- `summary.json` - wall-clock FPS, frame count, run config
- `env.json` - Python and package versions, OS release, kernel, device model,
  CPU governors/frequencies, SHA-256 of the models, run config
