#!/usr/bin/env python3
"""
Background SoC telemetry sampler for the long-term (endurance) test.

Runs ON THE DEVICE, independent of whichever pipeline is under load, so all
variants (Python / C++ / GStreamer) get one identical, comparable telemetry
stream. Pure stdlib - no venv, no numpy - so it starts instantly under the
system python3 while the pipeline owns the venv.

It samples every INTERVAL seconds until it receives SIGTERM/SIGINT (the host
orchestrator kills it at the end of each variant window) and appends one CSV
row per sample, flushing immediately so a kill never loses data:

  iso           wall-clock ISO-8601 (host aligns the ChargerLAB power .db to this)
  epoch         unix seconds (float)
  elapsed_s     seconds since this sampler started
  temp_c        SoC temperature, same thermal-zone priority as common.read_temp_c
  freq_khz_max  highest per-core scaling_cur_freq  (drops => thermal throttling)
  freq_khz_min  lowest  per-core scaling_cur_freq
  cpu_pct       whole-machine CPU utilisation over the last interval (/proc/stat)

Config via env (the same style as LOOP_RUNS / RENDER in the other device
scripts), with sensible defaults:

  OUT=results/telemetry_longterm.csv   output CSV path (relative to this dir)
  INTERVAL=2.0                         seconds between samples
  TAG=""                               free label echoed to stdout on start
"""

import csv
import os
import signal
import time
from pathlib import Path

THERMAL_ZONE_PRIORITY = ("cpu0-thermal", "cpuss0-thermal", "cpu-thermal")

BASE_DIR = Path(__file__).parent.resolve()
OUT = BASE_DIR / os.environ.get("OUT", "results/telemetry_longterm.csv")
INTERVAL = float(os.environ.get("INTERVAL", "2.0"))
TAG = os.environ.get("TAG", "")

CSV_HEADER = ["iso", "epoch", "elapsed_s", "temp_c",
              "freq_khz_max", "freq_khz_min", "cpu_pct"]

_stop = False


def _handle_stop(signum, frame):
    global _stop
    _stop = True


def read_temp_c():
    """SoC temperature in °C (same zone priority as the benchmark's common.py)."""
    zones = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            zones[(zone / "type").read_text().strip()] = zone / "temp"
        except OSError:
            pass
    for name in THERMAL_ZONE_PRIORITY:
        if name in zones:
            try:
                return int(zones[name].read_text().strip()) / 1000.0
            except (OSError, ValueError):
                pass
    return 0.0


def read_freqs_khz():
    """(max, min) current per-core frequency in kHz. A throttling SoC clocks the
    online cores down, so a falling max is the direct throttle signal."""
    freqs = []
    for cpu_dir in Path("/sys/devices/system/cpu").glob("cpu[0-9]*"):
        try:
            freqs.append(int((cpu_dir / "cpufreq" / "scaling_cur_freq")
                             .read_text().strip()))
        except (OSError, ValueError):
            pass
    if not freqs:
        return 0, 0
    return max(freqs), min(freqs)


def read_cpu_jiffies():
    """(total, busy) jiffies for the whole machine from /proc/stat line 'cpu'."""
    for line in Path("/proc/stat").read_text().splitlines():
        if line.startswith("cpu "):
            vals = [int(v) for v in line.split()[1:9]]
            total = sum(vals)
            busy = total - vals[3] - vals[4]   # - idle - iowait
            return total, busy
    return 0, 0


def main():
    signal.signal(signal.SIGTERM, _handle_stop)
    signal.signal(signal.SIGINT, _handle_stop)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"[sampler] TAG={TAG!r} INTERVAL={INTERVAL}s -> {OUT}", flush=True)

    t0 = time.monotonic()
    prev_total, prev_busy = read_cpu_jiffies()

    with open(OUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        f.flush()
        while not _stop:
            # The interval also defines the CPU% averaging window.
            slept = 0.0
            while slept < INTERVAL and not _stop:
                time.sleep(min(0.25, INTERVAL - slept))
                slept += 0.25

            now = time.time()
            total, busy = read_cpu_jiffies()
            dt = total - prev_total
            cpu_pct = 100.0 * (busy - prev_busy) / dt if dt else 0.0
            prev_total, prev_busy = total, busy
            fmax, fmin = read_freqs_khz()

            writer.writerow([
                time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                f"{now:.3f}",
                round(time.monotonic() - t0, 3),
                round(read_temp_c(), 1),
                fmax, fmin,
                round(cpu_pct, 1),
            ])
            f.flush()

    print(f"[sampler] stopped ({TAG!r})", flush=True)


if __name__ == "__main__":
    main()
