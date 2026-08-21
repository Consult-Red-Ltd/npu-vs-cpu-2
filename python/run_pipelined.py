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

# SAVE_VIDEO=1 burns the person/helmet/vest boxes into an annotated mp4 (per
# input clip) under output/pipelined/ - for eyeballing that detection works.
SAVE_VIDEO = os.environ.get("SAVE_VIDEO", "0") == "1"
# TRACE=1 records, per frame, the wall-clock [start,end] each stage spent
# working on it (perf_counter), buffered in memory and dumped once to
# results/pipelined/trace.csv at the end. Feeds the pipeline Gantt
# (host/plot_gantt.py) that shows frames overlapping across stages. Off by
# default - pure timestamp bookkeeping, no effect on the measured stage times.
TRACE = os.environ.get("TRACE", "0") == "1"
QUEUE_SIZE = 8

BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent               # article-2 root: shared input/ + models/
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = BASE_DIR / "output" / "pipelined"
RESULTS_DIR = BASE_DIR / "results" / "pipelined"
MODEL_PERSON = ROOT / "models" / "foot_track_net-tflite-w8a8" / "foot_track_net.tflite"
MODEL_PPE = ROOT / "models" / "gear_guard_net-tflite-w8a8" / "gear_guard_net.tflite"


def stage_reader(videos, models, q_out):
    for video in videos:
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_id = 0
        while True:
            t_stage0 = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                break
            # Keep the full frame in BGR. Colour-convert only the small model
            # input here (and each PPE crop in stage_crop) - a BGR2RGB on the
            # tiny model-input tensor is ~60x cheaper than on the full 1080p
            # frame, and the result is numerically identical (resize is per-
            # channel, so resize-then-swap == swap-then-resize).
            with common.Timer() as t_prep:
                resized = cv2.resize(frame, (models.w1, models.h1))
                input_a = np.expand_dims(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB), 0)
            now = time.perf_counter()
            item = {
                "video": video.name, "fps": fps, "frame_id": frame_id,
                "t_read": now,
                "frame": frame, "input_a": input_a,
                "times": {"prep_person_ms": t_prep.elapsed_ms,
                          "infer_person_ms": 0.0, "post_person_ms": 0.0,
                          "prep_ppe_ms": 0.0, "infer_ppe_ms": 0.0,
                          "post_ppe_ms": 0.0},
            }
            if TRACE:
                item["trace"] = {"reader": (t_stage0, now)}
            q_out.put(item)
            frame_id += 1
        cap.release()
    q_out.put(None)


def stage_crop(models, q_in, q_out):
    while True:
        item = q_in.get()
        if item is None:
            q_out.put(None)
            return
        t_stage0 = time.perf_counter()
        with common.Timer() as t_detect:
            persons = common.detect_persons_arrays(*item.pop("person_raw"))
        item["times"]["post_person_ms"] += t_detect.elapsed_ms

        frame = item["frame"]
        orig_h, orig_w = frame.shape[:2]
        scale_x = orig_w / models.w1
        scale_y = orig_h / models.h1

        item["persons"] = []
        item["inputs_b"] = []
        for px1, py1, px2, py2, score in persons:
            x1 = max(0, int(px1 * scale_x))
            y1 = max(0, int(py1 * scale_y))
            x2 = min(orig_w, int(px2 * scale_x))
            y2 = min(orig_h, int(py2 * scale_y))
            with common.Timer() as t_prep:
                crop = frame[y1:y2, x1:x2]              # BGR crop
                if crop.size == 0:
                    continue
                padded, crop_scale, pad_w, pad_h = common.resize_and_pad(
                    crop, models.h2, models.w2)
                input_b = np.expand_dims(cv2.cvtColor(padded, cv2.COLOR_BGR2RGB), 0)
            item["times"]["prep_ppe_ms"] += t_prep.elapsed_ms
            item["persons"].append((x1, y1, x2, y2, score, crop_scale, pad_w, pad_h))
            item["inputs_b"].append(input_b)

        item["orig_w"], item["orig_h"] = orig_w, orig_h
        if not SAVE_VIDEO:
            item.pop("frame")
        if "trace" in item:
            item["trace"]["crop"] = (t_stage0, time.perf_counter())
        q_out.put(item)


def stage_writer(q_in, csv_writer, counters, trace_rows):
    video_writer = None
    current_video = None
    while True:
        item = q_in.get()
        if item is None:
            break
        t_stage0 = time.perf_counter()

        results = []
        for (x1, y1, x2, y2, score, crop_scale, pad_w, pad_h), raw in zip(
                item["persons"], item["ppe_raw"]):
            with common.Timer() as t_post:
                detections = common.detect_ppe_arrays(*raw)
                ppe_global = common.ppe_to_global(
                    detections, x1, y1, crop_scale, pad_w, pad_h,
                    item["orig_w"], item["orig_h"])
            item["times"]["post_ppe_ms"] += t_post.elapsed_ms
            results.append((x1, y1, x2, y2, score, ppe_global))

        times = item["times"]
        total_ms = sum(times.values())
        # End-to-end latency: wall-clock from the frame being read (enqueued at
        # stage 1) to leaving the pipeline here. In a 5-stage pipeline this is
        # higher than total_ms (queue waits + stage contention) even though
        # throughput doubles - the throughput-vs-latency trade-off.
        latency_ms = (time.perf_counter() - item["t_read"]) * 1000.0
        csv_writer.writerow([
            item["video"], item["frame_id"], len(results),
            f"{times['prep_person_ms']:.3f}", f"{times['infer_person_ms']:.3f}",
            f"{times['post_person_ms']:.3f}",
            f"{times['prep_ppe_ms']:.3f}", f"{times['infer_ppe_ms']:.3f}",
            f"{times['post_ppe_ms']:.3f}",
            f"{total_ms:.3f}", f"{common.read_temp_c():.1f}", f"{latency_ms:.3f}",
        ])

        if SAVE_VIDEO:
            if item["video"] != current_video:
                if video_writer is not None:
                    video_writer.release()
                video_writer = cv2.VideoWriter(
                    str(OUTPUT_DIR / item["video"]), cv2.VideoWriter_fourcc(*"mp4v"),
                    item["fps"], (item["orig_w"], item["orig_h"]))
                current_video = item["video"]
            frame = item["frame"]
            common.draw_detections(frame, results)
            video_writer.write(frame)
        counters["frames"] += 1

        if "trace" in item:
            item["trace"]["writer"] = (t_stage0, time.perf_counter())
            trace_rows.append((item["frame_id"], item["video"], item["trace"]))

    if video_writer is not None:
        video_writer.release()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_VIDEO:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Mode: 5-stage pipeline | Delegate: {common.DELEGATE}")
    models = common.Models(MODEL_PERSON, MODEL_PPE)

    config = {"mode": "pipelined", "delegate": common.DELEGATE,
              "htp_perf_mode": common.HTP_PERF_MODE, "save_video": SAVE_VIDEO,
              "queue_size": QUEUE_SIZE,
              "stages": ["reader+scale", "model1", "crop+prep", "model2", "writer"]}
    common.write_env(RESULTS_DIR / "env.json", config, [MODEL_PERSON, MODEL_PPE])

    telemetry = common.Telemetry()
    telemetry.start()

    # VIDEO=<path> runs just that one clip (e.g. a quick single-clip visual
    # check); otherwise process every clip in input/.
    videos = ([Path(os.environ["VIDEO"])] if os.environ.get("VIDEO")
              else sorted(INPUT_DIR.glob("*.mp4")))
    queues = [queue.Queue(maxsize=QUEUE_SIZE) for _ in range(4)]
    counters = {"frames": 0}
    trace_rows = []          # (frame_id, video, {stage: (start, end)}) when TRACE

    # `trace_label` lets the shared model stages tag their per-frame [start,end]
    # into item["trace"] (only when the reader seeded one, i.e. TRACE=1).
    threads = [
        threading.Thread(target=stage_reader, args=(videos, models, queues[0]), daemon=True),
        threading.Thread(target=pipeline.stage_model1,
                         args=(models, queues[0], queues[1],
                               "prep_person_ms", "infer_person_ms", "post_person_ms"),
                         kwargs={"trace_label": "model1"}, daemon=True),
        threading.Thread(target=stage_crop, args=(models, queues[1], queues[2]), daemon=True),
        threading.Thread(target=pipeline.stage_model2,
                         args=(models, queues[2], queues[3],
                               "prep_ppe_ms", "infer_ppe_ms", "post_ppe_ms"),
                         kwargs={"trace_label": "model2"}, daemon=True),
    ]

    t_start = time.perf_counter()
    for thread in threads:
        thread.start()

    with open(RESULTS_DIR / "frames.csv", "w", newline="") as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(common.CSV_HEADER)
        stage_writer(queues[3], csv_writer, counters, trace_rows)

    duration_s = time.perf_counter() - t_start
    for thread in threads:
        thread.join(timeout=5)
    telemetry.stop_and_save(RESULTS_DIR / "telemetry.csv")

    if TRACE and trace_rows:
        stages = ["reader", "model1", "crop", "model2", "writer"]
        with open(RESULTS_DIR / "trace.csv", "w", newline="") as tf:
            tw = csv.writer(tf)
            tw.writerow(["frame_id", "video"]
                        + [f"{s}_{edge}" for s in stages for edge in ("start", "end")])
            for fid, video, tr in trace_rows:
                row = [fid, video]
                for s in stages:
                    s0, s1 = tr.get(s, (None, None))
                    row += [f"{s0:.6f}" if s0 is not None else "",
                            f"{s1:.6f}" if s1 is not None else ""]
                tw.writerow(row)
        print(f"Trace: {len(trace_rows)} frames -> {RESULTS_DIR / 'trace.csv'}")
    summary = common.write_summary(RESULTS_DIR / "summary.json", "pipelined",
                                   counters["frames"], duration_s, extra=config)
    print(f"Done: {summary['frames']} frames in {summary['duration_s']}s "
          f"-> {summary['fps']} FPS")
    print(f"Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
