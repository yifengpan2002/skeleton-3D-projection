"""
player_detection_no_calib.py
-----------------------------
YOLO detection WITHOUT field filtering (no homography/calibration used).
Keeps ALL detections - players, refs, bench, crowd, everyone.

Use this to compare against the calibrated version.
"""

import cv2
import os
import csv
import numpy as np
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
VIDEO_PATH   = "data/fifa/ARG_CRO_220001 (1).mp4"
FRAMES_ROOT  = "data/frames"
OUTPUT_ROOT  = "data/detections"
MODEL_PATH   = "model\yolo11x.pt"
FRAME_STEP   = 1
CONF_THRESH  = 0.2
TARGET_CLASS = "person"
SAVE_FRAMES  = True
IMG_SIZE     = 1280
# ─────────────────────────────────────────


def detect_players_no_filtering(
    frame_paths: list[str],
    model_path: str,
    output_dir: str,
    conf: float = 0.25,
    target_class: str = "person",
    img_size: int = 1280,
    save_frames: bool = True,
) -> list[dict]:
    """
    Run YOLOv11 on each frame - NO FIELD FILTERING.
    Keeps ALL detections (players, refs, bench, crowd).
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[YOLO]   Loading '{model_path}' ...")
    model = YOLO(model_path)

    class_names   = model.names
    target_cls_id = next(
        (k for k, v in class_names.items() if v.lower() == target_class.lower()), None
    )
    if target_cls_id is None:
        raise ValueError(
            f"Class '{target_class}' not found. "
            f"Available: {list(class_names.values())}"
        )

    print(f"[YOLO]   Target class -> '{target_class}' (id={target_cls_id})")
    print(f"[YOLO]   ⚠️  NO FIELD FILTERING - keeping ALL detections")
    print(f"[YOLO]   Processing {len(frame_paths)} frames ...\n")

    all_records = []
    total_detections = 0

    for i, frame_path in enumerate(frame_paths):
        frame_stem = Path(frame_path).stem
        frame_name = Path(frame_path).name

        results = model.predict(
            source  = frame_path,
            conf    = conf,
            classes = [target_cls_id],
            imgsz   = img_size,
            verbose = False,
        )

        result = results[0]
        boxes  = result.boxes
        records_this_frame = []

        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence      = round(float(box.conf[0]), 4)
            total_detections += 1

            records_this_frame.append({
                "frame_path" : frame_path,
                "frame_index": i,
                "player_id"  : j,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "confidence" : confidence,
            })

        all_records.extend(records_this_frame)

        # Draw ALL bounding boxes
        if save_frames:
            frame_img = cv2.imread(frame_path)
            for rec in records_this_frame:
                cv2.rectangle(
                    frame_img,
                    (rec["x1"], rec["y1"]),
                    (rec["x2"], rec["y2"]),
                    color=(0, 255, 0),
                    thickness=2,
                )
                label = f"person {rec['confidence']:.2f}"
                cv2.putText(
                    frame_img, label,
                    (rec["x1"], rec["y1"] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
            cv2.imwrite(os.path.join(output_dir, frame_name), frame_img)

        if (i + 1) % 50 == 0 or (i + 1) == len(frame_paths):
            print(f"  [{i + 1}/{len(frame_paths)}] {frame_name} "
                  f"-> {len(records_this_frame)} detection(s)")

    print(f"\n[Total]  All detections: {total_detections}")
    print(f"         (includes players + refs + bench + crowd)")

    return all_records


def compile_video(
    annotated_dir: str,
    output_path: str,
    original_video: str,
    frame_step: int = 1,
) -> None:
    """Stitch annotated frames back into an MP4."""
    frame_files = sorted(Path(annotated_dir).glob("frame_*.jpg"))
    if not frame_files:
        print("[Video]  No annotated frames found — skipping.")
        return

    cap          = cv2.VideoCapture(original_video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    output_fps = max(1.0, original_fps / frame_step)

    sample = cv2.imread(str(frame_files[0]))
    h, w   = sample.shape[:2]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

    print(f"\n[Video]  Compiling {len(frame_files)} frames "
          f"-> '{output_path}' @ {output_fps:.2f} fps ...")
    for f in frame_files:
        writer.write(cv2.imread(str(f)))
    writer.release()
    print(f"[Video]  Done -> '{output_path}'\n")


def save_csv(records: list[dict], output_path: str) -> None:
    """Write detection records to CSV."""
    if not records:
        print("[CSV]    No detections to save.")
        return
    fieldnames = ["frame_path", "frame_index", "player_id",
                  "x1", "y1", "x2", "y2", "confidence"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"[CSV]    Saved {len(records)} rows -> '{output_path}'")


def print_summary(records: list[dict], frame_paths: list[str]) -> None:
    total  = len(records)
    frames = len({r["frame_index"] for r in records})
    avg    = total / len(frame_paths) if frame_paths else 0
    print("\n-----------------------------------------")
    print("  DETECTION SUMMARY (NO FILTERING)")
    print("-----------------------------------------")
    print(f"  Total frames processed : {len(frame_paths)}")
    print(f"  Frames with detections : {frames}")
    print(f"  Total detections       : {total}")
    print(f"  Avg detections / frame : {avg:.2f}")
    print("-----------------------------------------\n")


# ─────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    video_name   = Path(VIDEO_PATH).stem
    frames_dir   = os.path.join(FRAMES_ROOT, video_name)
    output_dir   = os.path.join(OUTPUT_ROOT, video_name + "_no_calib")
    output_video = os.path.join(output_dir, f"{video_name}_no_calib.mp4")

    print(f"[Setup]  Video      : {video_name}")
    print(f"[Setup]  Frames dir : {frames_dir}")
    print(f"[Setup]  Output dir : {output_dir}")
    print(f"[Setup]  Mode       : NO CALIBRATION (all detections kept)\n")

    # Load frames
    frame_paths = sorted(str(p) for p in Path(frames_dir).glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in '{frames_dir}'. "
            "Run extract_frames.py first."
        )
    print(f"[Frames] Loaded {len(frame_paths)} frames\n")

    # Detect WITHOUT filtering
    records = detect_players_no_filtering(
        frame_paths  = frame_paths,
        model_path   = MODEL_PATH,
        output_dir   = output_dir,
        conf         = CONF_THRESH,
        target_class = TARGET_CLASS,
        img_size     = IMG_SIZE,
        save_frames  = SAVE_FRAMES,
    )

    # Save results
    save_csv(records, os.path.join(output_dir, "detections_no_calib.csv"))
    print_summary(records, frame_paths)

    # Compile video
    compile_video(output_dir, output_video, VIDEO_PATH, FRAME_STEP)
    
    print("\n[Done] To compare with calibrated version:")
    print(f"       1. Run: python player_detection.py")
    print(f"       2. Compare videos:")
    print(f"          - NO calib:   {output_video}")
    print(f"          - WITH calib: data/detections/{video_name}/{video_name}_annotated.mp4")
