"""
cascade_detect.py - run the full-HW PPE cascade (foot_track + gear_guard on the
Hexagon NPU) and stream per-frame detections as JSONL, so a separate program can
parse it and raise a "person without protective gear" alert.

This is the maximally-GStreamer path: the SAME qti-ml cascade as run_cascade.sh,
we only tap the two qtimlpostprocess `text/x-raw` outputs via appsink instead of
overlaying them. The person stage (module=qpd) gives person boxes with an `id`;
the gear stage (module=gearguard) gives helmet/vest boxes tagged with `parent-id`
(= the person id) - so person<->PPE association is explicit, no spatial guessing.

Output: one JSON object per processed frame (newline-delimited = trivial to
stream-parse), e.g.

  {"frame":137,"pts_ns":5480000000,"w":1920,"h":1080,
   "persons":[{"id":256,"bbox":[889,587,134,402],"helmet":true,"vest":false,"ppe_ok":false}],
   "alert":true}

- bbox = [x,y,w,h] in full-frame PIXELS; helmet/vest = that class detected under
  the person's parent-id (already confidence-thresholded by the postprocess
  settings, gear=40); ppe_ok = helmet AND vest; alert = any person not ppe_ok.
- Frames with no persons emit "persons":[] (heartbeat).

Config via env (no CLI args, per repo convention):
  INPUT=input/1.mp4        video to run (default the repo clip)
  OUT=-                    "-" = stdout (default); or a path / named FIFO
                           (mkfifo /dev/shm/dets.fifo) for live consumption
  FRAME_W=1920 FRAME_H=1080   frame size for the pixel bbox conversion

Needs python3-gi + GStreamer (both present on the Rubik: Gst 1.24).
"""

import json
import os
import re
import sys
from pathlib import Path

import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

BASE = Path(__file__).parent.resolve()
ROOT = BASE.parent
INPUT = os.environ.get("INPUT") or str(ROOT / "input" / "1.mp4")
OUT = os.environ.get("OUT", "-")
FRAME_W = int(os.environ.get("FRAME_W", "1920"))
FRAME_H = int(os.environ.get("FRAME_H", "1080"))

FOOT = str(ROOT / "models" / "foot_track_net-tflite-w8a8" / "foot_track_net.tflite")
GEAR = str(ROOT / "models" / "gear_guard_net-tflite-w8a8" / "gear_guard_net.tflite")
L = str(BASE / "labels")
QNN = "libQnnTFLiteDelegate.so"
OPTS = "QNNExternalDelegate,backend_type=htp;"

# The postprocess text is a GstStructure serialized as a GValue with deep
# backslash-escaping that Gst.Structure.from_string refuses to parse. But the
# fields we need survive a plain de-escape (strip backslashes) + regex - the
# numbers/labels themselves are never mangled, only the delimiters.
DET_RE = re.compile(
    r"(person|helmet|vest),\s*id=\(uint\)(\d+),\s*confidence=\(double\)([\d.eE+-]+),"
    r"\s*color=\(uint\)\d+,\s*rectangle=\(float\)<\s*"
    r"([\d.eE+-]+),\s*([\d.eE+-]+),\s*([\d.eE+-]+),\s*([\d.eE+-]+)\s*>")
TS_RE = re.compile(r"timestamp=\(guint64\)(\d+)")
PARENT_RE = re.compile(r"parent-id=\(int\)(-?\d+)")


def _clean(raw: str) -> str:
    return raw.replace("\\", "")


def parse_persons(raw: str):
    """-> (ts, {id: (x,y,w,h) normalized})."""
    s = _clean(raw)
    ts_m = TS_RE.search(s)
    ts = int(ts_m.group(1)) if ts_m else None
    persons = {}
    for m in DET_RE.finditer(s):
        if m.group(1) != "person":
            continue
        pid = int(m.group(2))
        persons[pid] = tuple(float(m.group(i)) for i in (4, 5, 6, 7))
    return ts, persons


def parse_gear(raw: str):
    """-> (ts, parent_id, has_helmet, has_vest). One gear buffer = one person crop."""
    s = _clean(raw)
    ts_m = TS_RE.search(s)
    ts = int(ts_m.group(1)) if ts_m else None
    par_m = PARENT_RE.search(s)
    parent = int(par_m.group(1)) if par_m else None
    helmet = vest = False
    for m in DET_RE.finditer(s):
        if m.group(1) == "helmet":
            helmet = True
        elif m.group(1) == "vest":
            vest = True
    return ts, parent, helmet, vest


class Joiner:
    """Buffers person + gear samples by frame timestamp and emits one JSON line
    per frame once the frame is complete. Person text is produced upstream of
    gear, so for a frame the person sample always arrives before its gear
    samples; a frame is complete when we've collected `expected` per-person gear
    entries (or the frame had no persons). A watermark flush handles stragglers."""

    def __init__(self, emit):
        self._emit = emit
        self._frames = {}   # ts -> {"persons":{}, "ppe":{}, "expected":int|None, "seq":int}
        self._frame_no = 0
        self._max_ts = 0
        self._emitted_ts = -1   # timestamps are monotonic; drop anything <= this
                                # (guards against a duplicate first buffer)

    def _f(self, ts):
        return self._frames.setdefault(
            ts, {"persons": {}, "ppe": {}, "expected": None})

    def on_person(self, raw):
        ts, persons = parse_persons(raw)
        if ts is None or ts <= self._emitted_ts:
            return
        self._max_ts = max(self._max_ts, ts)
        f = self._f(ts)
        f["persons"] = persons
        f["expected"] = len(persons)   # authoritative crop count for this frame
        self._drain()

    def on_gear(self, raw):
        ts, parent, helmet, vest = parse_gear(raw)
        if ts is None or ts <= self._emitted_ts:
            return
        self._max_ts = max(self._max_ts, ts)
        f = self._f(ts)
        if parent is not None:
            f["ppe"][parent] = (helmet, vest)
        self._drain()

    def _complete(self, f):
        return (f["expected"] is not None
                and len(f["ppe"]) >= f["expected"])

    def _drain(self, flush_all=False):
        # Emit in timestamp order any frames that are complete (or, on flush /
        # watermark, everything old enough that later frames have already begun).
        for ts in sorted(self._frames):
            f = self._frames[ts]
            straggler = (not flush_all) and (ts < self._max_ts) and (f["expected"] is not None)
            if flush_all or self._complete(f) or straggler:
                self._emit_frame(ts, f)
                del self._frames[ts]
            else:
                # oldest not ready and not a straggler -> wait for more
                if not flush_all:
                    break

    def _emit_frame(self, ts, f):
        self._emitted_ts = ts
        persons = []
        for pid, (x, y, w, h) in sorted(f["persons"].items()):
            helmet, vest = f["ppe"].get(pid, (False, False))
            persons.append({
                "id": pid,
                "bbox": [round(x * FRAME_W), round(y * FRAME_H),
                         round(w * FRAME_W), round(h * FRAME_H)],
                "helmet": helmet, "vest": vest, "ppe_ok": helmet and vest,
            })
        obj = {
            "frame": self._frame_no,
            "pts_ns": ts,
            "w": FRAME_W, "h": FRAME_H,
            "persons": persons,
            "alert": any(not p["ppe_ok"] for p in persons),
        }
        self._frame_no += 1
        self._emit(json.dumps(obj, separators=(",", ":")))

    def flush(self):
        self._drain(flush_all=True)


def build_pipeline():
    return Gst.parse_launch(
        f'filesrc location="{INPUT}" ! qtdemux ! h264parse ! v4l2h264dec ! tee name=t0 '
        f't0. ! queue ! mux1. '
        f't0. ! queue ! qtimlvconverter ! '
        f'  qtimltflite model="{FOOT}" delegate=external external-delegate-path="{QNN}" '
        f'  external-delegate-options="{OPTS}" ! '
        f'  qtimlpostprocess module=qpd results=10 labels="{L}/foot_track_net.txt" '
        f'  settings="{L}/foot_track_net_settings.json" ! text/x-raw ! tee name=ptext '
        f'  ptext. ! queue ! mux1. '
        f'  ptext. ! queue ! appsink name=person emit-signals=true max-buffers=30 drop=false sync=false '
        f'qtimetamux name=mux1 ! queue ! qtimlvconverter mode=roi-batch-cumulative ! '
        f'  qtimltflite model="{GEAR}" delegate=external external-delegate-path="{QNN}" '
        f'  external-delegate-options="{OPTS}" ! '
        f'  qtimlpostprocess module=gearguard results=10 labels="{L}/gear_guard_net.txt" '
        f'  settings="{L}/gear_guard_net_settings.json" ! text/x-raw ! '
        f'  appsink name=gear emit-signals=true max-buffers=30 drop=false sync=false')


def _pull_text(sink):
    sample = sink.emit("pull-sample")
    if sample is None:
        return None
    buf = sample.get_buffer()
    ok, minfo = buf.map(Gst.MapFlags.READ)
    if not ok:
        return None
    try:
        return bytes(minfo.data).decode("utf-8", "replace")
    finally:
        buf.unmap(minfo)


def main():
    Gst.init(None)
    out = sys.stdout if OUT == "-" else open(OUT, "w", buffering=1, encoding="utf-8")
    loop = GLib.MainLoop()

    def emit(line):
        try:
            out.write(line + "\n")
            out.flush()
        except BrokenPipeError:
            loop.quit()   # consumer went away -> stop cleanly instead of stalling

    joiner = Joiner(emit)
    pipe = build_pipeline()

    pipe.get_by_name("person").connect(
        "new-sample", lambda s: (joiner.on_person(_pull_text(s) or ""), Gst.FlowReturn.OK)[1])
    pipe.get_by_name("gear").connect(
        "new-sample", lambda s: (joiner.on_gear(_pull_text(s) or ""), Gst.FlowReturn.OK)[1])

    bus = pipe.get_bus()
    bus.add_signal_watch()

    def on_msg(_bus, msg):
        if msg.type == Gst.MessageType.EOS:
            try:
                joiner.flush()
            except Exception as e:
                sys.stderr.write(f"[cascade_detect] flush error: {e}\n")
            loop.quit()
        elif msg.type == Gst.MessageType.ERROR:
            err, dbg = msg.parse_error()
            sys.stderr.write(f"[cascade_detect] ERROR: {err} :: {dbg}\n")
            loop.quit()
    bus.connect("message", on_msg)

    sys.stderr.write(f"[cascade_detect] INPUT={INPUT} OUT={OUT} -> streaming JSONL\n")
    pipe.set_state(Gst.State.PLAYING)
    try:
        loop.run()
    finally:
        pipe.set_state(Gst.State.NULL)
        if out is not sys.stdout:
            out.close()


if __name__ == "__main__":
    main()
