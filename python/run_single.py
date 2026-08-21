import csv
import os
import time
from pathlib import Path

import cv2

import common

SAVE_VIDEO = False

BASE_DIR = Path(__file__).parent.resolve()
ROOT = BASE_DIR.parent
INPUT_DIR = ROOT / "input"
OUTPUT_DIR = BASE_DIR / "output" / "single"
RESULTS_DIR = BASE_DIR / "results" / "single"
MODEL_PERSON = ROOT / "models" / "foot_track_net-tflite-w8a8" / "foot_track_net.tflite"
MODEL_PPE = ROOT / "models" / "gear_guard_net-tflite-w8a8" / "gear_guard_net.tflite"


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if SAVE_VIDEO:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Mode: single-threaded | Delegate: {common.DELEGATE}")
    models = common.Models(MODEL_PERSON, MODEL_PPE)

    config = {"mode": "single", "delegate": common.DELEGATE,
              "htp_perf_mode": common.HTP_PERF_MODE, "save_video": SAVE_VIDEO}
    common.write_env(RESULTS_DIR / "env.json", config, [MODEL_PERSON, MODEL_PPE])

    telemetry = common.Telemetry()
    telemetry.start()

    total_frames = 0
    t_start = time.perf_counter()

    with open(RESULTS_DIR / "frames.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(common.CSV_HEADER)

        # VIDEO=<path> runs just that one clip (matches run_pipelined/benchmark);
        # otherwise process every clip in input/.
        videos = ([Path(os.environ["VIDEO"])] if os.environ.get("VIDEO")
                  else sorted(INPUT_DIR.glob("*.mp4")))
        for video in videos:
            print(f"Processing: {video.name}")
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            video_writer = None
            if SAVE_VIDEO:
                video_writer = cv2.VideoWriter(
                    str(OUTPUT_DIR / video.name), cv2.VideoWriter_fourcc(*"mp4v"),
                    fps, (orig_w, orig_h))

            frame_id = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                t_read = time.perf_counter()
                results, times = common.analyze_frame(models, frame, orig_w, orig_h)

                # End-to-end latency. Single-threaded, so it tracks total compute
                # closely - kept for column parity with the pipelined variant,
                # where latency and throughput diverge.
                latency_ms = (time.perf_counter() - t_read) * 1000.0
                writer.writerow([
                    video.name, frame_id, len(results),
                    f"{times['prep_person_ms']:.3f}", f"{times['infer_person_ms']:.3f}",
                    f"{times['post_person_ms']:.3f}",
                    f"{times['prep_ppe_ms']:.3f}", f"{times['infer_ppe_ms']:.3f}",
                    f"{times['post_ppe_ms']:.3f}",
                    f"{times['total_ms']:.3f}", f"{common.read_temp_c():.1f}",
                    f"{latency_ms:.3f}",
                ])

                if video_writer is not None:
                    common.draw_detections(frame, results)
                    video_writer.write(frame)
                frame_id += 1

            cap.release()
            if video_writer is not None:
                video_writer.release()
            total_frames += frame_id

    duration_s = time.perf_counter() - t_start
    telemetry.stop_and_save(RESULTS_DIR / "telemetry.csv")
    summary = common.write_summary(RESULTS_DIR / "summary.json", "single",
                                   total_frames, duration_s, extra=config)
    print(f"Done: {summary['frames']} frames in {summary['duration_s']}s "
          f"-> {summary['fps']} FPS")
    print(f"Results in {RESULTS_DIR}")


if __name__ == "__main__":
    main()
