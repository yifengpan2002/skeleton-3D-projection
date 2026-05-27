"""
player_detection.py
--------------------
Runs YOLOv11 player detection on extracted frames, saves annotated
frames, a CSV, and an output video.

Output:
    data/detections/<video_name>/
        ├── frame_XXXXXX.jpg          ← annotated frames
        ├── detections.csv            ← all detection records
        └── <video_name>_annotated.mp4

Can be run standalone or imported by main.py.
"""

import cv2
import os
import csv
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
VIDEO_PATH   = "data/clips-NRL/Adam_Doueihi-Tigers_v_Eels_NRL_R6_2023.mkv"
FRAMES_DIR   = "data/annotated_rugby_player"
OUTPUT_DIR   = "outputs/player_detect/annotated_player"
MODEL_PATH   = "model\yolo11x.pt"
FRAME_STEP   = 1
CONF_THRESH  = 0.35
TARGET_CLASS = "person"
SAVE_FRAMES  = True
IMG_SIZE     = 1280
# ─────────────────────────────────────────


# ─────────────────────────────────────────
#  DETECTION
# ─────────────────────────────────────────

def detect_players(
    frame_paths: list[str],
    model_path: str,
    output_dir: str,
    conf: float = 0.25,
    target_class: str = "person",
    img_size: int = 1280,
    save_frames: bool = True,
) -> list[dict]:
    """
    Run YOLOv11 on each frame and save detections.
    Returns a list of detection records.
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
    print(f"[YOLO]   Processing {len(frame_paths)} frames ...\n")

    all_records = []
    total_raw   = 0
    total_kept  = 0

    for i, frame_path in enumerate(frame_paths):
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
            total_raw      += 1

            # Basic box-size filter: remove tiny crowd/background detections
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 8 or box_h < 25:
                continue

            total_kept += 1
            records_this_frame.append({
                "frame_path" : frame_path,
                "frame_index": i,
                "player_id"  : j,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "confidence" : confidence,
            })

        all_records.extend(records_this_frame)

        # Draw bounding boxes and save annotated frame
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
                label = f"player {rec['confidence']:.2f}"
                cv2.putText(
                    frame_img, label,
                    (rec["x1"], rec["y1"] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )
            cv2.imwrite(os.path.join(output_dir, frame_name), frame_img)

        if (i + 1) % 50 == 0 or (i + 1) == len(frame_paths):
            print(f"  [{i + 1}/{len(frame_paths)}] {frame_name} "
                  f"-> {len(records_this_frame)} player(s)")

    print(f"\n[Filter] Raw detections : {total_raw}")
    print(f"[Filter] Kept           : {total_kept}")
    print(f"[Filter] Discarded      : {total_raw - total_kept}")

    return all_records


# ─────────────────────────────────────────
#  VIDEO COMPILATION
# ─────────────────────────────────────────

def compile_video(
    annotated_dir: str,
    output_path: str,
    original_video: str,
    frame_step: int = 1,
) -> None:
    """Stitch annotated frames back into an MP4 at the original FPS."""
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


# ─────────────────────────────────────────
#  CSV + SUMMARY
# ─────────────────────────────────────────

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
    print("  DETECTION SUMMARY")
    print("-----------------------------------------")
    print(f"  Total frames processed : {len(frame_paths)}")
    print(f"  Frames with detections : {frames}")
    print(f"  Total players          : {total}")
    print(f"  Avg players / frame    : {avg:.2f}")
    print("-----------------------------------------\n")


# ─────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    video_name   = Path(VIDEO_PATH).stem
    output_dir   = OUTPUT_DIR
    output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")

    print(f"[Setup]  Video      : {video_name}")
    print(f"[Setup]  Frames dir : {FRAMES_DIR}")
    print(f"[Setup]  Output dir : {output_dir}\n")

    # Load existing frames
    frame_paths = sorted(str(p) for p in Path(FRAMES_DIR).glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in '{FRAMES_DIR}'."
        )
    print(f"[Frames] Loaded {len(frame_paths)} frames from '{FRAMES_DIR}'\n")

    # Detect
    records = detect_players(
        frame_paths  = frame_paths,
        model_path   = MODEL_PATH,
        output_dir   = output_dir,
        conf         = CONF_THRESH,
        target_class = TARGET_CLASS,
        img_size     = IMG_SIZE,
        save_frames  = SAVE_FRAMES,
    )

    # Save results
    save_csv(records, os.path.join(output_dir, "detections.csv"))
    print_summary(records, frame_paths)

    # Compile video
    compile_video(output_dir, output_video, VIDEO_PATH, FRAME_STEP)