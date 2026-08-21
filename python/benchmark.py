import csv
import os
import queue
import threading
import time
from pathlib import Path

import cv2
import numpy as np

import common
import pipeline

WARMUP_RUNS = 10
MAX_FRAMES = 0
QUEUE_SIZE = 8
# Replay the input video this many times back-to-back. Default 1 keeps the
# short perf run; the power/thermal sessions set LOOP_RUNS in the environment
# (see host/measure_03.py) so the run is long enough for a steady power trace
# and a realistic SoC temperature, not dominated by warm-up / model load.
LOOP_RUNS = int(os.environ.get("LOOP_RUNS", "1"))
# If RUN_SECONDS > 0 it OVERRIDES LOOP_RUNS: replay the clip in-process until
# this many seconds of wall-clock elapse. The long-term/endurance session sets
# it so ONE invocation holds the load for the whole window - the models stay
# loaded (no per-invocation reload blip), giving a clean continuous thermal/CPU
# trace instead of the sawtooth caused by host-side process re-invocation.
RUN_SECONDS = float(os.environ.get("RUN_SECONDS", "0"))

BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
# VIDEO is overridable so the long-term session can feed one long looped clip
# from /dev/shm (RAM) instead of the short repo clip - same knob for all three
# variants (python/C++/gstreamer). Default stays input/1.mp4 for short runs.
VIDEO = Path(os.environ.get("VIDEO") or (ROOT / "input" / "1.mp4"))
# Own subdir so a shared python/results/ holds run_single -> single/,
# run_pipelined -> pipelined/, and this benchmark -> benchmark/ side by side.
RESULTS_DIR = BASE_DIR / "results" / "benchmark"
MODEL_PERSON = ROOT / "models" / "foot_track_net-tflite-w8a8" / "foot_track_net.tflite"
MODEL_PPE = ROOT / "models" / "gear_guard_net-tflite-w8a8" / "gear_guard_net.tflite"

CSV_HEADER = ["frame_id", "num_persons_detected",
              "decode_time_ms", "prep_A_ms", "infer_A_ms", "post_A_ms",
              "prep_B_ms", "infer_B_ms", "post_B_ms", "total_frame_ms",
              "cpu_usage_percent", "per_core_cpu_percent", "ram_usage_mb", "temp_C"]

TIME_KEYS = ["prep_A_ms", "infer_A_ms", "post_A_ms",
             "prep_B_ms", "infer_B_ms", "post_B_ms"]


def warmup(interpreter, runs):
    detail = interpreter.get_input_details()[0]
    interpreter.set_tensor(detail["index"], np.zeros(detail["shape"], dtype=np.uint8))
    for _ in range(runs):
        interpreter.invoke()


def stage_reader(models, q_out, open_acc):
    frame_id = 0
    loop = 0
    t_start = time.perf_counter()

    def time_up():
        return RUN_SECONDS > 0 and (time.perf_counter() - t_start) >= RUN_SECONDS

    while not time_up():
        # Accumulate capture-open time so it can be excluded from the wall FPS
        # (symmetry with the C++ twin - a fair sprint excludes decoder init).
        t_open = time.perf_counter()
        cap = cv2.VideoCapture(str(VIDEO))
        open_acc["ms"] += (time.perf_counter() - t_open) * 1000.0
        # Time cap is checked PER FRAME (not just per file loop) so it stops on
        # the dot whether VIDEO is the short clip (re-looped) or the long
        # /dev/shm clip (read once, mid-file).
        while (MAX_FRAMES <= 0 or frame_id < MAX_FRAMES) and not time_up():
            t0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            decode_ms = (time.perf_counter() - t0) * 1000.0
            # Colour-convert only the small model input, not the full frame; the
            # BGR frame is carried forward for PPE crops (converted per-crop in
            # stage_crop). NB: decode_time_ms no longer includes the full-frame
            # cvtColor - it is now pure cap.read() decode time.
            with common.Timer() as t_prep:
                resized = cv2.resize(frame, (models.w1, models.h1))
                input_a = np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), 0)
            q_out.put({
                "frame_id": frame_id, "frame": frame, "input_a": input_a,
                "times": {"decode_time_ms": decode_ms,
                          "prep_A_ms": t_prep.elapsed_ms, "infer_A_ms": 0.0,
                          "post_A_ms": 0.0, "prep_B_ms": 0.0, "infer_B_ms": 0.0,
                          "post_B_ms": 0.0},
            })
            frame_id += 1
        cap.release()
        loop += 1
        if MAX_FRAMES > 0 and frame_id >= MAX_FRAMES:
            break
        if RUN_SECONDS <= 0 and loop >= LOOP_RUNS:
            break
    q_out.put(None)


def stage_crop(models, q_in, q_out):
    while True:
        item = q_in.get()
        if item is None:
            q_out.put(None)
            return
        with common.Timer() as t_detect:
            persons = common.detect_persons_arrays(*item.pop("person_raw"))
        item["times"]["post_A_ms"] += t_detect.elapsed_ms

        frame = item.pop("frame")
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / models.w1
        scale_y = orig_h / models.h1

        item["num_persons"] = 0
        item["inputs_b"] = []
        for px1, py1, px2, py2, _score in persons:
            x1 = max(0, int(px1 * scale_x))
            y1 = max(0, int(py1 * scale_y))
            x2 = min(orig_w, int(px2 * scale_x))
            y2 = min(orig_h, int(py2 * scale_y))
            with common.Timer() as t_prep:
                crop = frame[y1:y2, x1:x2]             # BGR crop
                if crop.size == 0:
                    continue
                padded, _, _, _ = common.resize_and_pad(crop, models.h2, models.w2)
                input_b = np.expand_dims(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), 0)
            item["times"]["prep_B_ms"] += t_prep.elapsed_ms
            item["num_persons"] += 1
            item["inputs_b"].append(input_b)
        q_out.put(item)


def stage_collect(q_in, telemetry, rows):
    while True:
        item = q_in.get()
        if item is None:
            return
        for raw in item.pop("ppe_raw"):
            with common.Timer() as t_post:
                common.detect_ppe_arrays(*raw)
            item["times"]["post_B_ms"] += t_post.elapsed_ms

        times = item["times"]
        total_ms = sum(times[key] for key in TIME_KEYS)
        sample = telemetry.rows[-1] if telemetry.rows else [0, 0.0, "", 0.0, 0.0]
        rows.append([
            item["frame_id"], item["num_persons"],
            f"{times['decode_time_ms']:.3f}",
            f"{times['prep_A_ms']:.3f}", f"{times['infer_A_ms']:.3f}",
            f"{times['post_A_ms']:.3f}",
            f"{times['prep_B_ms']:.3f}", f"{times['infer_B_ms']:.3f}",
            f"{times['post_B_ms']:.3f}", f"{total_ms:.3f}",
            sample[1], sample[2], sample[3], sample[4],
        ])


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Implementation: python | Delegate: {common.DELEGATE} | "
          f"Video: {VIDEO.name} | Mode: pipelined")
    models = common.Models(MODEL_PERSON, MODEL_PPE)

    config = {"implementation": "python", "mode": "pipelined", "delegate": common.DELEGATE,
              "htp_perf_mode": common.HTP_PERF_MODE, "video": VIDEO.name,
              "warmup_runs": WARMUP_RUNS, "max_frames": MAX_FRAMES,
              "loop_runs": LOOP_RUNS, "run_seconds": RUN_SECONDS,
              "queue_size": QUEUE_SIZE,
              "stages": ["reader+scale", "model1", "crop+prep", "model2", "collector"]}
    common.write_env(RESULTS_DIR / "env_python.json", config, [MODEL_PERSON, MODEL_PPE])

    print(f"Warmup: {WARMUP_RUNS} runs per model ...")
    warmup(models.person, WARMUP_RUNS)
    warmup(models.ppe, WARMUP_RUNS)

    telemetry = common.Telemetry()
    telemetry.start()

    queues = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(4)]
    open_acc = {"ms": 0.0}       # capture-open time, excluded from wall FPS
    threads = [
        threading.Thread(target=stage_reader, args=(models, queues[0], open_acc), daemon=True),
        threading.Thread(target=pipeline.stage_model1,
                         args=(models, queues[0], queues[1],
                               "prep_A_ms", "infer_A_ms", "post_A_ms"), daemon=True),
        threading.Thread(target=stage_crop, args=(models, queues[1], queues[2]), daemon=True),
        threading.Thread(target=pipeline.stage_model2,
                         args=(models, queues[2], queues[3],
                               "prep_B_ms", "infer_B_ms", "post_B_ms"), daemon=True),
    ]

    rows = []
    t_bench_start = time.perf_counter()
    for thread in threads:
        thread.start()
    stage_collect(queues[3], telemetry, rows)
    # Exclude capture-open time (reader has finished by now) so FPS is
    # steady-state processing - matches the C++ twin's fair-sprint timing.
    duration_s = time.perf_counter() - t_bench_start - open_acc["ms"] / 1000.0
    for thread in threads:
        thread.join(timeout=5)

    telemetry.stop_and_save(RESULTS_DIR / "telemetry_python.csv")

    with open(RESULTS_DIR / "python_metrics.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(rows)

    summary = common.write_summary(RESULTS_DIR / "summary_python.json", "pipelined",
                                   len(rows), duration_s, extra=config)
    print(f"Done: {summary['frames']} frames in {summary['duration_s']}s "
          f"-> {summary['fps']} FPS")
    print(f"Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
