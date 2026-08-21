import csv
import hashlib
import json
import os
import platform
import sys
import threading
import time
from datetime import datetime
from importlib import metadata
from pathlib import Path

import cv2
import numpy as np
from ai_edge_litert import interpreter as litert

# NPU config - identical across every run script, so it lives here (single
# source of truth) rather than being redeclared and threaded through each caller.
DELEGATE = "hexagon"
HTP_PERF_MODE = 2

PERSON_SCORE_THRESH = 0.76
PERSON_IOU_THRESH = 0.5
PPE_SCORE_THRESH = 0.40
PPE_IOU_THRESH = 0.5
STRIDE = 4
TOP_K = 1000

COLOR_PERSON = (255, 200, 0)
COLOR_HELMET = (0, 165, 255)
COLOR_VEST = (0, 255, 0)

THERMAL_ZONE_PRIORITY = ("cpu0-thermal", "cpuss0-thermal", "cpu-thermal")

CSV_HEADER = ["video", "frame_id", "num_persons",
              "prep_person_ms", "infer_person_ms", "post_person_ms",
              "prep_ppe_ms", "infer_ppe_ms", "post_ppe_ms",
              "total_ms", "temp_c", "latency_ms"]

TELEMETRY_HEADER = ["elapsed_s", "cpu_percent", "per_core_percent", "rss_mb", "temp_c"]


class Timer:
    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed_ms = (time.perf_counter() - self._start) * 1000.0
        return False


def load_interpreter(model_path, delegate=DELEGATE, htp_perf_mode=HTP_PERF_MODE):
    # LiteRT (ai-edge-litert) replaces the tensorflow.lite runtime. The NPU path
    # (delegate == "hexagon") loads the Qualcomm QNN delegate onto the Hexagon
    # HTP, which compiles the whole graph - so we do NOT set num_threads (it only
    # tunes LiteRT's built-in CPU XNNPACK delegate, which the NPU path never uses;
    # matches article-1, which dropped the thread knob from the headline runs).
    delegates = []
    if delegate == "hexagon":
        # https://docs.qualcomm.com/nav/home/options.html?product=1601111740010412
        options = {"backend_type": "htp", "htp_performance_mode": str(htp_perf_mode)}
        try:
            delegates.append(litert.load_delegate("libQnnTFLiteDelegate.so", options=options))
        except Exception as e:
            print(f"WARN: Hexagon delegate unavailable ({e}), falling back to CPU.")
    interpreter = litert.Interpreter(
        model_path=str(model_path),
        experimental_delegates=delegates or None,
    )
    interpreter.allocate_tensors()
    return interpreter


class Models:
    def __init__(self, model_person, model_ppe, delegate=DELEGATE, htp_perf_mode=HTP_PERF_MODE):
        self.person = load_interpreter(model_person, delegate, htp_perf_mode)
        self.ppe = load_interpreter(model_ppe, delegate, htp_perf_mode)
        self.person_input = self.person.get_input_details()[0]
        self.ppe_input = self.ppe.get_input_details()[0]
        self.h1, self.w1 = self.person_input["shape"][1], self.person_input["shape"][2]
        self.h2, self.w2 = self.ppe_input["shape"][1], self.ppe_input["shape"][2]


def get_output(interpreter, name, dequantize=True):
    detail = next(d for d in interpreter.get_output_details() if d["name"] == name)
    tensor = interpreter.get_tensor(detail["index"])
    scale, zero_point = detail["quantization"]
    if dequantize and scale > 0.0:
        tensor = (tensor.astype(np.float32) - zero_point) * scale
    return tensor


def get_output_raw(interpreter, name):
    """Raw (un-dequantized) output tensor plus its (scale, zero_point)."""
    detail = next(d for d in interpreter.get_output_details() if d["name"] == name)
    scale, zero_point = detail["quantization"]
    return interpreter.get_tensor(detail["index"]), scale, zero_point


def extract_person_outputs(interpreter):
    # Return the RAW quantized tensors + their quant params; detect_persons_arrays
    # dequantizes only the few selected cells. Dequantizing the whole bbox tensor
    # here (230k values, to use ~8) was ~0.4 ms/frame of pure waste on the Rubik
    # A78 (host/bench_dequant.py: 3.3x on this step). Bonus: uint8 (not float32)
    # crosses the pipeline queue - 4x less to hand between threads.
    hm_raw, hm_s, hm_zp = get_output_raw(interpreter, "heatmap")
    bb_raw, bb_s, bb_zp = get_output_raw(interpreter, "bbox")
    return hm_raw, bb_raw, (hm_s, hm_zp, bb_s, bb_zp)


def detect_persons_arrays(heatmap, tlrb, quant):
    # Threshold on the RAW uint8 heatmap (scores > T  <=>  raw > T/scale + zp, for
    # scale > 0), then dequantize ONLY the selected scores and their bbox cells.
    # Boxes are built with numpy fancy-indexing (no Python loop over peaks) and
    # NMS runs via cv2.dnn.NMSBoxes - same tool the PPE path uses.
    hm_s, hm_zp, bb_s, bb_zp = quant
    raw = heatmap[0, :, :, 1]
    thr = (PERSON_SCORE_THRESH / hm_s + hm_zp) if hm_s > 0.0 else PERSON_SCORE_THRESH
    ys, xs = np.nonzero(raw > thr)
    if ys.shape[0] == 0:
        return []
    sc = raw[ys, xs].astype(np.float32)
    if hm_s > 0.0:
        sc = (sc - hm_zp) * hm_s
    if sc.shape[0] > TOP_K:
        order = np.argsort(-sc)[:TOP_K]
        ys, xs, sc = ys[order], xs[order], sc[order]
    off = tlrb[0, ys, xs, 4:8].astype(np.float32)    # (n,4): left, top, right, bottom
    if bb_s > 0.0:
        off = (off - bb_zp) * bb_s
    x1 = (xs - off[:, 0]) * STRIDE
    y1 = (ys - off[:, 1]) * STRIDE
    x2 = (xs + off[:, 2]) * STRIDE
    y2 = (ys + off[:, 3]) * STRIDE
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    rects = np.column_stack([x1, y1, x2 - x1, y2 - y1]).tolist()   # xywh for cv2
    keep = cv2.dnn.NMSBoxes(rects, sc.astype(float).tolist(),
                            PERSON_SCORE_THRESH, PERSON_IOU_THRESH)
    if len(keep) == 0:
        return []
    return [(boxes[i, 0], boxes[i, 1], boxes[i, 2], boxes[i, 3], float(sc[i]))
            for i in np.asarray(keep).flatten()]


def detect_persons(interpreter):
    return detect_persons_arrays(*extract_person_outputs(interpreter))


def extract_ppe_outputs(interpreter):
    return (get_output(interpreter, "boxes")[0],
            get_output(interpreter, "scores")[0],
            get_output(interpreter, "class_idx", dequantize=False)[0])


def detect_ppe_arrays(boxes, scores, classes):
    valid = scores > PPE_SCORE_THRESH
    boxes, scores, classes = boxes[valid], scores[valid], classes[valid]
    if len(boxes) == 0:
        return []
    rects = [
        [int(x1), int(y1), int(max(0, x2 - x1)), int(max(0, y2 - y1))]
        for x1, y1, x2, y2 in boxes
    ]
    keep = cv2.dnn.NMSBoxes(rects, scores.astype(float).tolist(), PPE_SCORE_THRESH, PPE_IOU_THRESH)
    if len(keep) == 0:
        return []
    return [
        (boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], int(classes[i]), float(scores[i]))
        for i in np.asarray(keep).flatten()
    ]


def detect_ppe(interpreter):
    return detect_ppe_arrays(*extract_ppe_outputs(interpreter))


def ppe_to_global(detections, x1, y1, crop_scale, pad_w, pad_h, orig_w, orig_h):
    results = []
    for bx1, by1, bx2, by2, class_id, score in detections:
        gx1 = max(0, int(x1 + (bx1 - pad_w) / crop_scale))
        gy1 = max(0, int(y1 + (by1 - pad_h) / crop_scale))
        gx2 = min(orig_w, int(x1 + (bx2 - pad_w) / crop_scale))
        gy2 = min(orig_h, int(y1 + (by2 - pad_h) / crop_scale))
        results.append((gx1, gy1, gx2, gy2, class_id, score))
    return results


def resize_and_pad(image, target_h, target_w):
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    canvas[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
    return canvas, scale, pad_w, pad_h


def analyze_frame(models, frame, orig_w, orig_h):
    # `frame` is BGR; colour is converted on the small model inputs only, never
    # on the full frame (see the note in run_pipelined.stage_reader).
    times = {}
    with Timer() as t:
        resized = cv2.resize(frame, (models.w1, models.h1))
        models.person.set_tensor(models.person_input["index"],
                                 np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), 0))
    times["prep_person_ms"] = t.elapsed_ms
    with Timer() as t:
        models.person.invoke()
    times["infer_person_ms"] = t.elapsed_ms
    with Timer() as t:
        persons = detect_persons(models.person)
    times["post_person_ms"] = t.elapsed_ms

    times["prep_ppe_ms"] = 0.0
    times["infer_ppe_ms"] = 0.0
    times["post_ppe_ms"] = 0.0

    scale_x = orig_w / models.w1
    scale_y = orig_h / models.h1
    results = []
    for px1, py1, px2, py2, person_score in persons:
        x1 = max(0, int(px1 * scale_x))
        y1 = max(0, int(py1 * scale_y))
        x2 = min(orig_w, int(px2 * scale_x))
        y2 = min(orig_h, int(py2 * scale_y))

        with Timer() as t:
            crop = frame[y1:y2, x1:x2]             # BGR crop
            if crop.size == 0:
                continue
            padded, crop_scale, pad_w, pad_h = resize_and_pad(crop, models.h2, models.w2)
            models.ppe.set_tensor(models.ppe_input["index"],
                                  np.expand_dims(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), 0))
        times["prep_ppe_ms"] += t.elapsed_ms

        with Timer() as t:
            models.ppe.invoke()
        times["infer_ppe_ms"] += t.elapsed_ms

        with Timer() as t:
            detections = detect_ppe(models.ppe)
        times["post_ppe_ms"] += t.elapsed_ms

        ppe_global = ppe_to_global(detections, x1, y1, crop_scale,
                                   pad_w, pad_h, orig_w, orig_h)
        results.append((x1, y1, x2, y2, person_score, ppe_global))

    times["total_ms"] = sum(v for k, v in times.items() if k.endswith("_ms"))
    return results, times


def draw_box(frame, x1, y1, x2, y2, color, label):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (x1 + 4, max(14, y1 - 6)), cv2.FONT_HERSHEY_DUPLEX,
                0.45, color, 1, cv2.LINE_AA)


def draw_detections(frame, results):
    for x1, y1, x2, y2, person_score, ppe_global in results:
        draw_box(frame, x1, y1, x2, y2, COLOR_PERSON, f"Person {person_score:.0%}")
        for gx1, gy1, gx2, gy2, class_id, _score in ppe_global:
            color = COLOR_HELMET if class_id == 0 else COLOR_VEST
            name = "Helmet" if class_id == 0 else "Vest"
            draw_box(frame, gx1, gy1, gx2, gy2, color, name)


_temp_path = None          # cached /sys/.../temp file for the preferred zone
_temp_resolved = False     # None is a valid "no zone found" result, so gate on this


def _resolve_temp_path():
    """Glob /sys/class/thermal ONCE and return the temp file for the highest
    -priority zone (or None). read_temp_c caches this so the hot path is a single
    file read, not a directory glob + N type-file reads on every call - the
    pipeline samples it per frame, so the glob storm was pure CPU overhead."""
    zones = {}
    for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
        try:
            zones[(zone / "type").read_text().strip()] = zone / "temp"
        except OSError:
            pass
    for name in THERMAL_ZONE_PRIORITY:
        if name in zones:
            return zones[name]
    return None


def read_temp_c():
    global _temp_path, _temp_resolved
    if not _temp_resolved:
        _temp_path = _resolve_temp_path()
        _temp_resolved = True
    if _temp_path is None:
        return 0.0
    try:
        return int(_temp_path.read_text().strip()) / 1000.0
    except (OSError, ValueError):
        return 0.0


class Telemetry(threading.Thread):
    def __init__(self, interval_s=0.2):
        super().__init__(daemon=True)
        self.interval_s = interval_s
        self.rows = []
        self._stop_event = threading.Event()
        self._t0 = None

    @staticmethod
    def _read_jiffies():
        total, per_core = (0, 0), []
        for line in Path("/proc/stat").read_text().splitlines():
            if not line.startswith("cpu"):
                break
            parts = line.split()
            vals = [int(v) for v in parts[1:9]]
            entry = (sum(vals), sum(vals) - vals[3] - vals[4])
            if parts[0] == "cpu":
                total = entry
            else:
                per_core.append(entry)
        return total, per_core

    @staticmethod
    def _read_rss_mb():
        parts = Path("/proc/self/statm").read_text().split()
        return int(parts[1]) * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)

    def run(self):
        self._t0 = time.perf_counter()
        prev_total, prev_cores = self._read_jiffies()
        while not self._stop_event.wait(self.interval_s):
            total, cores = self._read_jiffies()
            dt = total[0] - prev_total[0]
            cpu_pct = 100.0 * (total[1] - prev_total[1]) / dt if dt else 0.0
            core_pcts = [
                100.0 * (c[1] - p[1]) / (c[0] - p[0]) if (c[0] - p[0]) else 0.0
                for c, p in zip(cores, prev_cores)
            ]
            prev_total, prev_cores = total, cores
            self.rows.append([
                round(time.perf_counter() - self._t0, 3),
                round(cpu_pct, 1),
                "|".join(f"{p:.0f}" for p in core_pcts),
                round(self._read_rss_mb(), 1),
                round(read_temp_c(), 1),
            ])

    def stop_and_save(self, path):
        self._stop_event.set()
        self.join(timeout=5)
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(TELEMETRY_HEADER)
            writer.writerows(self.rows)


def _read_file(path):
    try:
        return Path(path).read_text(errors="ignore").strip().strip("\x00")
    except OSError:
        return ""


def _cpu_freq_info():
    info = {}
    for cpu_dir in sorted(Path("/sys/devices/system/cpu").glob("cpu[0-9]*")):
        governor = _read_file(cpu_dir / "cpufreq" / "scaling_governor")
        max_freq = _read_file(cpu_dir / "cpufreq" / "cpuinfo_max_freq")
        if governor or max_freq:
            info[cpu_dir.name] = {"governor": governor, "max_freq_khz": max_freq}
    return info


def write_env(path, config, model_paths):
    env = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "kernel": platform.release(),
        "libc": "-".join(platform.libc_ver()),
        "device_model": _read_file("/proc/device-tree/model"),
        "os_release": _read_file("/etc/os-release"),
        "cpu": _cpu_freq_info(),
        "packages": {d.metadata["Name"]: d.version for d in metadata.distributions()},
        "models": {
            Path(m).name: hashlib.sha256(Path(m).read_bytes()).hexdigest()
            for m in model_paths
        },
        "config": config,
    }
    Path(path).write_text(json.dumps(env, indent=2), encoding="utf-8")


def write_summary(path, mode, frames, duration_s, extra=None):
    summary = {
        "mode": mode,
        "finished": datetime.now().isoformat(timespec="seconds"),
        "frames": frames,
        "duration_s": round(duration_s, 2),
        "fps": round(frames / duration_s, 2) if duration_s else 0.0,
    }
    if extra:
        summary.update(extra)
    Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
