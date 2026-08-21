# long-term - endurance (thermal + power) test

The rest of article 2 measures *peak* throughput. This one measures what happens
when you **keep** a variant under load. The point: the hand-tuned C++ pipeline is
the raw-speed ceiling, but it also works the CPU hardest - under a sustained run
it heats up, the SoC throttles and wall power stays high. The full-HW GStreamer
cascade offloads almost everything to the NPU/GPU, so it holds a **flat, cool,
steady** long run. "C++ is fastest; GStreamer is the one you'd leave running."

## What runs

Each variant is held under load for `DURATION_MIN` (default 15), with a cooldown
to a common baseline temperature in between. All three reuse the pipelines that
already live elsewhere in the article - nothing is duplicated here:

| Variant | Pipeline | Reused from |
|---------|----------|-------------|
| `python`    | pipelined, NPU              | `../python/benchmark.py` |
| `cpp`       | C++, GStreamer HW decode, NPU | `../cpp/run_cpp.sh` (`DECODE_VARIANT=gstreamer`) |
| `gstreamer` | full-HW cascade, NPU        | `../gstreamer/run_cascade.sh` (`RENDER=0`) |

`sampler.py` is the only code here: a tiny, dependency-free (stdlib) background
sampler that runs **on the device** independently of whichever pipeline is under
load, so every variant gets one identical, comparable telemetry stream.

## Method

- **Load** - the host re-invokes the pipeline back to back until `DURATION_MIN`
  elapses. Each invocation replays the clip several times (`INNER`), so one
  invocation is ~75–90 s and also yields one throughput sample → FPS-over-time.
  The brief per-invocation QNN re-init is the price of a uniform, per-variant FPS
  series; the background sampler runs *continuously* across the gaps, so the
  thermal/power trace is unaffected.
- **Telemetry** - `sampler.py` logs every `SAMPLE_INTERVAL_S` (default 2 s):
  SoC temperature (`cpu0-thermal` priority, same as the benchmarks), max/min
  per-core `scaling_cur_freq` (a falling max is the direct throttle signal),
  whole-machine CPU%, with **wall-clock** timestamps so the Power-Z trace aligns.
- **Power** - one continuous recording. Each variant's window is sliced out of
  the single `.db` by its `[start,end]` host timestamps plus the ChargerLAB DST
  offset (auto-detected, `plot_01.chargerlab_offset`).
