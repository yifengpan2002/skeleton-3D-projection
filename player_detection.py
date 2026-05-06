"""
Player Detection Pipeline using YOLOv11
----------------------------------------
Step 1: Split a video clip into frames
Step 2: Run YOLOv11 on each frame to detect players
Step 3: Save annotated frames and a summary CSV
Step 4: Compile annotated frames back into a video

Frames and detections are stored in subfolders named after the video file.
"""

import cv2
import os
import csv
from pathlib import Path
from ultralytics import YOLO


# ─────────────────────────────────────────
#  CONFIGURATION  (edit these as needed)
# ─────────────────────────────────────────
VIDEO_PATH   = "data\clips-NRL\Adam_Doueihi-Tigers_v_Eels_NRL_R6_2023.mkv"
FRAMES_ROOT  = "data/frames"          # Subfolder per video created automatically
OUTPUT_ROOT  = "data/detections"      # Subfolder per video created automatically
MODEL_PATH   = "yolo11x.pt"           # YOLOv11 model weights (n/s/m/l/x)
FRAME_STEP   = 1                      # Extract every N-th frame (1 = every frame)
CONF_THRESH  = 0.6                # Minimum confidence to keep a detection
TARGET_CLASS = "person"               # COCO class to treat as "player"
SAVE_FRAMES  = True                   # Save annotated frames to disk (needed for video)
IMG_SIZE     = 1920                   # Inference resolution (larger = better accuracy)
# ─────────────────────────────────────────


def get_video_name(video_path: str) -> str:
    """Extract the video filename without extension to use as subfolder name."""
    return Path(video_path).stem


def extract_frames(video_path: str, output_dir: str, frame_step: int = 1) -> list[str]:
    """
    Extract frames from a video and save them as JPEG images.
    Returns a sorted list of saved frame file paths.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Video]  {video_path}")
    print(f"         {total_frames} frames | {fps:.2f} fps | {width}x{height}")
    print(f"         Extracting every {frame_step} frame(s)...")

    saved_paths = []
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_step == 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_paths.append(filename)

        frame_idx += 1

    cap.release()
    print(f"[Frames] Saved {len(saved_paths)} frames -> '{output_dir}'\n")
    return sorted(saved_paths)


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
    Run YOLOv11 inference on each frame and return detection records.
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[Model]  Loading weights from '{model_path}' ...")
    model = YOLO(model_path)

    class_names   = model.names
    target_cls_id = next(
        (k for k, v in class_names.items() if v.lower() == target_class.lower()), None
    )
    if target_cls_id is None:
        raise ValueError(f"Class '{target_class}' not found in model. Available: {list(class_names.values())}")

    print(f"[Model]  Target class -> '{target_class}' (id={target_cls_id})")
    print(f"[Detect] Processing {len(frame_paths)} frames ...\n")

    all_records = []

    for i, frame_path in enumerate(frame_paths):
        results = model.predict(
            source    = frame_path,
            conf      = conf,
            classes   = [target_cls_id],
            imgsz     = img_size,
            verbose   = False,
        )

        result     = results[0]
        boxes      = result.boxes
        frame_name = Path(frame_path).name

        records_this_frame = []
        for j, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            confidence      = round(float(box.conf[0]), 4)

            records_this_frame.append({
                "frame_path" : frame_path,
                "frame_index": i,
                "player_id"  : j,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "confidence" : confidence,
            })

        all_records.extend(records_this_frame)

        if save_frames:
            annotated = result.plot()
            out_path  = os.path.join(output_dir, frame_name)
            cv2.imwrite(out_path, annotated)

        if (i + 1) % 50 == 0 or (i + 1) == len(frame_paths):
            print(f"  [{i + 1}/{len(frame_paths)}] {frame_name} -> {len(records_this_frame)} player(s) detected")

    return all_records


def compile_video(
    annotated_dir: str,
    output_path: str,
    original_video: str,
    frame_step: int = 1,
) -> None:
    """
    Reconstruct an MP4 video from the annotated frames.
    """
    frame_files = sorted(Path(annotated_dir).glob("frame_*.jpg"))
    if not frame_files:
        print("[Video]  No annotated frames found — skipping video compilation.")
        return

    cap = cv2.VideoCapture(original_video)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    output_fps = max(1.0, original_fps / frame_step)

    sample = cv2.imread(str(frame_files[0]))
    h, w   = sample.shape[:2]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, output_fps, (w, h))

    print(f"\n[Video]  Compiling {len(frame_files)} frames -> '{output_path}' @ {output_fps:.2f} fps ...")

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        writer.write(frame)

    writer.release()
    print(f"[Video]  Done! Annotated video saved -> '{output_path}'\n")


def save_csv(records: list[dict], output_path: str) -> None:
    """Write detection records to a CSV file."""
    if not records:
        print("[CSV]    No detections to save.")
        return

    fieldnames = ["frame_path", "frame_index", "player_id", "x1", "y1", "x2", "y2", "confidence"]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    print(f"\n[CSV]    Saved {len(records)} detection rows -> '{output_path}'")


def print_summary(records: list[dict], frame_paths: list[str]) -> None:
    total_players = len(records)
    frames_with_detections = len({r["frame_index"] for r in records})
    avg = total_players / len(frame_paths) if frame_paths else 0

    print("\n-----------------------------------------")
    print("  DETECTION SUMMARY")
    print("-----------------------------------------")
    print(f"  Total frames processed : {len(frame_paths)}")
    print(f"  Frames with detections : {frames_with_detections}")
    print(f"  Total player boxes     : {total_players}")
    print(f"  Avg players / frame    : {avg:.2f}")
    print("-----------------------------------------\n")


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────
if __name__ == "__main__":
    # Derive subfolder name from video filename
    video_name = get_video_name(VIDEO_PATH)
    frames_dir   = os.path.join(FRAMES_ROOT, video_name)
    output_dir   = os.path.join(OUTPUT_ROOT, video_name)
    output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")

    print(f"[Setup]  Video name  : {video_name}")
    print(f"[Setup]  Frames dir  : {frames_dir}")
    print(f"[Setup]  Output dir  : {output_dir}")
    print(f"[Setup]  Output video: {output_video}\n")

    # 1. Extract frames
    frame_paths = extract_frames(VIDEO_PATH, frames_dir, FRAME_STEP)

    # 2. Detect players on every frame
    records = detect_players(
        frame_paths  = frame_paths,
        model_path   = MODEL_PATH,
        output_dir   = output_dir,
        conf         = CONF_THRESH,
        target_class = TARGET_CLASS,
        img_size     = IMG_SIZE,
        save_frames  = SAVE_FRAMES,
    )

    # 3. Save CSV results + print summary
    save_csv(records, os.path.join(output_dir, "detections.csv"))
    print_summary(records, frame_paths)

    # 4. Compile annotated frames back into a video
    compile_video(
        annotated_dir  = output_dir,
        output_path    = output_video,
        original_video = VIDEO_PATH,
        frame_step     = FRAME_STEP,
    )