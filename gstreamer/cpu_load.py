"""
Background CPU + RAM meter for the gst-launch passes (which have no telemetry
of their own). Snapshots /proc/stat at start and again on SIGTERM, and tracks
the peak RSS of the gst-launch process throughout, then prints JSON:
{ cpu_total_pct, per_core_pct, n_cores, ram_peak_mb }.

cpu_total_pct is 0-100% of the whole machine (100% = all cores busy); ram_peak_mb
is the peak resident set of the gst-launch process (same metric as experiments
01-03, which read /proc/self/statm). Stdlib only - no venv needed.

  python3 cpu_load.py > cpu.json &   PID=$!
  ...run the pipeline...
  kill -TERM $PID; wait $PID
"""
import glob
import json
import os
import signal
import time

PAGE_MB = os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)


def gst_rss_mb():
    """Sum resident MB of all gst-launch processes (comm is truncated to 15c)."""
    total_pages = 0
    for d in glob.glob("/proc/[0-9]*"):
        try:
            with open(d + "/comm") as f:
                if "gst-launch" not in f.read():
                    continue
            with open(d + "/statm") as f:
                total_pages += int(f.read().split()[1])
        except (OSError, ValueError, IndexError):
            continue
    return total_pages * PAGE_MB


def read():
    agg, cores = None, []
    for line in open("/proc/stat"):
        if not line.startswith("cpu"):
            break
        p = line.split()
        v = list(map(int, p[1:9]))      # user nice system idle iowait irq softirq steal
        total = sum(v)
        busy = total - v[3] - v[4]      # exclude idle + iowait
        if p[0] == "cpu":
            agg = (total, busy)
        else:
            cores.append((total, busy))
    return agg, cores


running = [True]
signal.signal(signal.SIGTERM, lambda *_: running.__setitem__(0, False))

a0, c0 = read()
ram_peak = 0.0
while running[0]:
    r = gst_rss_mb()
    if r > ram_peak:
        ram_peak = r
    time.sleep(0.1)
a1, c1 = read()


def pct(x0, x1):
    d = x1[0] - x0[0]
    return round(100.0 * (x1[1] - x0[1]) / d, 1) if d > 0 else 0.0


n = min(len(c0), len(c1))
print(json.dumps({
    "cpu_total_pct": pct(a0, a1),
    "per_core_pct": [pct(c0[i], c1[i]) for i in range(n)],
    "n_cores": n,
    "ram_peak_mb": round(ram_peak, 1),
}))
