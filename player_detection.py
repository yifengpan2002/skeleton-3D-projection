"""
player_detection.py
--------------------
Runs YOLOv11 player detection on extracted frames, filters detections
to on-field players only using homography matrices from tvcalib_preprocess.py,
saves annotated frames, a CSV, and an output video.

Output:
    data/detections/<video_name>/
        ├── frame_XXXXXX.jpg          ← annotated frames
        ├── detections.csv            ← all on-field detection records
        └── <video_name>_annotated.mp4

Can be run standalone or imported by main.py.
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
VIDEO_PATH   = "data/clips-NRL/Adam_Doueihi-Tigers_v_Eels_NRL_R6_2023.mkv"
FRAMES_ROOT  = "data/frames"
OUTPUT_ROOT  = "data/detections"
MODEL_PATH   = "model\yolo11x.pt"
FRAME_STEP   = 1
CONF_THRESH  = 0.35
TARGET_CLASS = "person"
SAVE_FRAMES  = True
IMG_SIZE     = 1280

# NRL field dimensions in metres
FIELD_LENGTH = 100.0
FIELD_WIDTH  =  68.0
FIELD_MARGIN =   2.0    # padding so sideline players aren't wrongly discarded
# ─────────────────────────────────────────

FIELD_POLYGON = np.array([
    [-FIELD_MARGIN,                -FIELD_MARGIN],
    [FIELD_LENGTH + FIELD_MARGIN,  -FIELD_MARGIN],
    [FIELD_LENGTH + FIELD_MARGIN,   FIELD_WIDTH + FIELD_MARGIN],
    [-FIELD_MARGIN,                 FIELD_WIDTH + FIELD_MARGIN],
], dtype=np.float32)


# Image-space field polygon.
# These are pixel coordinates on the 1280x720 frame.
# You MUST adjust these points after checking your first output frame.
USE_IMAGE_FIELD_MASK = True

FIELD_POLYGON_IMAGE = np.array([
    [40, 250],      # top-left visible field
    [1240, 230],    # top-right visible field
    [1275, 715],    # bottom-right visible field
    [10, 715],      # bottom-left visible field
], dtype=np.int32)

# ─────────────────────────────────────────
#  LOAD HOMOGRAPHIES
# ─────────────────────────────────────────

def load_homographies_npz(npz_path: str) -> dict[str, np.ndarray]:
    """
    Load homography matrices saved by tvcalib_preprocess.py.
    Returns { 'frame_000042': np.ndarray (3,3) or None, ... }
    NaN-filled rows (failed calibration) are returned as None.
    """
    if not Path(npz_path).exists():
        raise FileNotFoundError(
            f"NPZ not found: '{npz_path}'\n"
            "Run tvcalib_preprocess.py first."
        )

    data         = np.load(npz_path, allow_pickle=False)
    frame_names  = data["frame_names"]
    homographies = data["homographies"]

    result = {}
    for name, H in zip(frame_names, homographies):
        result[str(name)] = None if np.any(np.isnan(H)) else H.astype(np.float32)

    valid = sum(1 for v in result.values() if v is not None)
    print(f"[NPZ]    Loaded {len(result)} homographies "
          f"({valid} valid, {len(result) - valid} failed) from '{npz_path}'")
    return result


# ─────────────────────────────────────────
#  FIELD FILTERING
# ─────────────────────────────────────────

def is_on_field(px: int, py: int, H: np.ndarray, polygon: np.ndarray) -> bool:
    """
    Return True if image pixel (px, py) maps inside the field polygon.
    Uses the player's feet (bottom-centre of bounding box) for accuracy.
    """
    try:
        pt        = np.array([[[px, py]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, H)
        fx, fy    = float(projected[0, 0, 0]), float(projected[0, 0, 1])
        result    = cv2.pointPolygonTest(polygon, (fx, fy), measureDist=False)
        return result >= 0
    except Exception:
        return False


def is_inside_image_field(px: int, py: int) -> bool:
    """
    Return True if image pixel (px, py) is inside the manually defined
    field polygon in image coordinates.
    """
    result = cv2.pointPolygonTest(
        FIELD_POLYGON_IMAGE,
        (float(px), float(py)),
        measureDist=False
    )
    return result >= 0
# ─────────────────────────────────────────
#  DETECTION
# ─────────────────────────────────────────

def detect_players(
    frame_paths: list[str],
    model_path: str,
    output_dir: str,
    homography_map: dict[str, np.ndarray],
    conf: float = 0.25,
    target_class: str = "person",
    img_size: int = 1280,
    save_frames: bool = True,
) -> list[dict]:
    """
    Run YOLOv11 on each frame and filter detections to on-field players only.
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

    all_records   = []
    total_raw     = 0
    total_kept    = 0
    no_homography = 0

    for i, frame_path in enumerate(frame_paths):
        frame_stem = Path(frame_path).stem
        frame_name = Path(frame_path).name
        H          = homography_map.get(frame_stem)

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

            # Bottom-centre = player's feet
            feet_x = (x1 + x2) // 2
            feet_y = y2

            if H is None:
                on_field = True     # no homography — keep as fallback
                no_homography += 1
            else:
                on_field = is_on_field(feet_x, feet_y, H, FIELD_POLYGON)

            # if not on_field:
            #     continue
            # Basic box-size filter: remove tiny crowd/background detections
            box_w = x2 - x1
            box_h = y2 - y1

            if box_w < 8 or box_h < 25:
                continue

            # Bottom-centre = player's feet
            feet_x = (x1 + x2) // 2
            feet_y = y2

            # Use image-space field mask instead of TVCalib homography
            if USE_IMAGE_FIELD_MASK:
                on_field = is_inside_image_field(feet_x, feet_y)
            else:
                if H is None:
                    on_field = True
                    no_homography += 1
                else:
                    on_field = is_on_field(feet_x, feet_y, H, FIELD_POLYGON)

            if not on_field:
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

        # Draw on-field bounding boxes and save annotated frame
        if save_frames:
            frame_img = cv2.imread(frame_path)
            if USE_IMAGE_FIELD_MASK:
                cv2.polylines(
                    frame_img,
                    [FIELD_POLYGON_IMAGE],
                    isClosed=True,
                    color=(255, 0, 0),
                    thickness=2
                )
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
                  f"-> {len(records_this_frame)} on-field player(s)")

    print(f"\n[Filter] Raw detections  : {total_raw}")
    print(f"[Filter] Kept (on-field) : {total_kept}")
    print(f"[Filter] Discarded       : {total_raw - total_kept}")
    if no_homography > 0:
        print(f"[Filter] No-homography frames (kept all): {no_homography}")

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
    print(f"  Total on-field players : {total}")
    print(f"  Avg players / frame    : {avg:.2f}")
    print("-----------------------------------------\n")


# ─────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    from pathlib import Path

    video_name   = Path(VIDEO_PATH).stem
    frames_dir   = os.path.join(FRAMES_ROOT, video_name)
    output_dir   = os.path.join(OUTPUT_ROOT, video_name)
    output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    npz_path     = os.path.join(output_dir, "homographies.npz")

    print(f"[Setup]  Video      : {video_name}")
    print(f"[Setup]  Frames dir : {frames_dir}")
    print(f"[Setup]  Output dir : {output_dir}")
    print(f"[Setup]  NPZ file   : {npz_path}\n")

    # Load homographies
    homography_map = load_homographies_npz(npz_path)
    # Load homographies only if using TVCalib filtering
    if USE_IMAGE_FIELD_MASK:
        homography_map = {}
        print("[NPZ]    Skipping homography loading because USE_IMAGE_FIELD_MASK=True\n")
    else:
        homography_map = load_homographies_npz(npz_path)
    # Load existing frames
    frame_paths = sorted(str(p) for p in Path(frames_dir).glob("frame_*.jpg"))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in '{frames_dir}'. "
            "Run extract_frames.py first."
        )
    print(f"[Frames] Loaded {len(frame_paths)} frames from '{frames_dir}'\n")

    # Detect + filter
    records = detect_players(
        frame_paths    = frame_paths,
        model_path     = MODEL_PATH,
        output_dir     = output_dir,
        homography_map = homography_map,
        conf           = CONF_THRESH,
        target_class   = TARGET_CLASS,
        img_size       = IMG_SIZE,
        save_frames    = SAVE_FRAMES,
    )

    # Save results
    save_csv(records, os.path.join(output_dir, "detections.csv"))
    print_summary(records, frame_paths)

    # Compile video
    compile_video(output_dir, output_video, VIDEO_PATH, FRAME_STEP)