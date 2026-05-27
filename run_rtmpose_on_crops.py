"""
run_rtmpose_on_crops.py
------------------------
Runs RTMW (RTMPose Wholebody, 133 keypoints) on each cropped player
image produced by extract_player_crops.py. Each crop holds a single
player, so RTMW has no ambiguity — cleaner keypoints than running on
the whole frame.

Keypoints are mapped from COCO-Wholebody (133) to the 15-joint format
used in the worldpose evaluation pipeline (includes toes via the
wholebody foot keypoints).

Expected input structure:
    outputs/player_detect/annotated_player/
        ├── frame_000000/
        │   ├── player_0.jpg
        │   └── player_1.jpg
        ├── frame_000020/
        │   └── ...

Output structure:
    outputs/player_detect/annotated_player/
        ├── frame_000000/                       (crops)
        ├── frame_000000_rtmpose/               ← NEW
        │   ├── player_0.jpg                    (rendered skeleton)
        │   ├── player_1.jpg
        │   └── frame_000000_keypoints.csv      (one CSV per frame)
        ├── frame_000020_rtmpose/
        │   └── ...

Dependencies
------------
  pip install rtmlib
"""

import sys
import csv
from pathlib import Path

import numpy as np
import cv2
import torch


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
ROOT_DIR     = "outputs/player_detect/annotated_player"

# RTMW model size
MODE         = "balanced"   # lightweight | balanced | performance
BACKEND      = "onnxruntime"

# Drawing
KPT_THRESH   = 0.3          # confidence threshold for drawing a keypoint
RENDER_FULL_WHOLEBODY = True  # if True, draws all 133 keypoints (face/hands too);
                              # if False, draws only the 15-joint skeleton
# ─────────────────────────────────────────


# ─────────────────────────────────────────
#  Joint mapping (from worldpose_rtmw_wholebody_eval.py)
# ─────────────────────────────────────────
WHOLEBODY133_TO_OUR15 = [
    0,    # 0:  nose
    6,    # 1:  Rshoulder
    5,    # 2:  Lshoulder
    8,    # 3:  Relbow
    7,    # 4:  Lelbow
    10,   # 5:  Rwrist
    9,    # 6:  Lwrist
    12,   # 7:  Rhip
    11,   # 8:  Lhip
    14,   # 9:  Rknee
    13,   # 10: Lknee
    16,   # 11: Rankle
    15,   # 12: Lankle
    20,   # 13: Rfoot  (right_big_toe)
    17,   # 14: Lfoot  (left_big_toe)
]

JOINT_NAMES = [
    "nose", "Rshoulder", "Lshoulder", "Relbow", "Lelbow",
    "Rwrist", "Lwrist", "Rhip", "Lhip", "Rknee",
    "Lknee", "Rankle", "Lankle", "Rfoot", "Lfoot",
]

# Skeleton edges between the 15 joints (for rendering)
SKELETON_15 = [
    (0, 1), (0, 2),                # nose -> shoulders
    (1, 3), (3, 5),                # right arm
    (2, 4), (4, 6),                # left arm
    (1, 7), (2, 8), (7, 8),        # torso
    (7, 9), (9, 11), (11, 13),     # right leg
    (8, 10), (10, 12), (12, 14),   # left leg
]


# ─────────────────────────────────────────
#  Rendering
# ─────────────────────────────────────────

def draw_15_skeleton(img, kpts15, scores15, thr=0.3):
    """Draw the 15-joint skeleton on top of an image (in place)."""
    H, W = img.shape[:2]
    for i, (x, y) in enumerate(kpts15):
        if scores15[i] < thr:
            continue
        if not (0 <= x < W and 0 <= y < H):
            continue
        cv2.circle(img, (int(x), int(y)), 3, (0, 255, 255), -1)

    for a, b in SKELETON_15:
        if scores15[a] < thr or scores15[b] < thr:
            continue
        pa = (int(kpts15[a, 0]), int(kpts15[a, 1]))
        pb = (int(kpts15[b, 0]), int(kpts15[b, 1]))
        cv2.line(img, pa, pb, (0, 255, 0), 2)
    return img


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def main():
    root_dir = Path(ROOT_DIR)
    if not root_dir.exists():
        sys.exit(f"[error] Root dir not found: {root_dir}")

    # Find crop subfolders (skip *_rtmpose, *_openpose, *_hrnet folders)
    frame_folders = sorted(
        p for p in root_dir.iterdir()
        if p.is_dir()
        and not any(p.name.endswith(s) for s in ("_rtmpose", "_openpose", "_hrnet"))
        and any(p.glob("*.jpg"))
    )
    if not frame_folders:
        sys.exit(f"[error] No crop subfolders in '{root_dir}'")

    # Load RTMW model once
    try:
        from rtmlib import Wholebody
    except ImportError:
        sys.exit("[error] rtmlib not installed. Run: pip install rtmlib")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup]  Root dir     : {root_dir}")
    print(f"[Setup]  Mode         : {MODE}")
    print(f"[Setup]  Device       : {device}")
    print(f"[Setup]  Crop folders : {len(frame_folders)}")
    print(f"[Setup]  Loading RTMW Wholebody ...\n")

    wholebody = Wholebody(mode=MODE, backend=BACKEND, device=device)

    # Optional: rtmlib's built-in drawer for the full 133-keypoint render
    draw_skeleton = None
    if RENDER_FULL_WHOLEBODY:
        try:
            from rtmlib import draw_skeleton as _ds
            draw_skeleton = _ds
        except ImportError:
            print("[warn] rtmlib.draw_skeleton not found — falling back to 15-joint render")

    # CSV header
    csv_header = ["player_id", "person_index"]
    for kp in JOINT_NAMES:
        csv_header += [f"{kp}_x", f"{kp}_y", f"{kp}_c"]

    succeeded     = 0
    failed        = 0
    total_imgs    = 0
    total_rows    = 0

    for i, folder in enumerate(frame_folders, start=1):
        out_dir = root_dir / f"{folder.name}_rtmpose"
        out_dir.mkdir(parents=True, exist_ok=True)

        crop_files = sorted(folder.glob("*.jpg"))
        total_imgs += len(crop_files)
        rows = []

        print(f"  [{i}/{len(frame_folders)}] {folder.name} ({len(crop_files)} crops) "
              f"-> '{out_dir.name}'")

        for crop_path in crop_files:
            img = cv2.imread(str(crop_path))
            if img is None:
                continue
            stem = crop_path.stem    # e.g. "player_0"

            try:
                keypoints, scores = wholebody(img)
            except Exception as e:
                print(f"    [error] RTMW failed on {crop_path.name}: {e}")
                row = [stem, -1] + [""] * (len(JOINT_NAMES) * 3)
                rows.append(row)
                continue

            kpts = np.asarray(keypoints, dtype=np.float32)
            scs  = np.asarray(scores,    dtype=np.float32)

            if kpts.size == 0:
                row = [stem, -1] + [""] * (len(JOINT_NAMES) * 3)
                rows.append(row)
                # still save the un-annotated crop so the folder is complete
                cv2.imwrite(str(out_dir / crop_path.name), img)
                continue

            if kpts.ndim == 2:    # single-person edge case
                kpts = kpts[None]
                scs  = scs[None]

            # Use the first detected person (one player per crop)
            kpts133 = kpts[0]                          # (133, 2)
            scs133  = scs[0]                           # (133,)
            kpts15  = kpts133[WHOLEBODY133_TO_OUR15]   # (15, 2)
            scs15   = scs133[WHOLEBODY133_TO_OUR15]    # (15,)

            # Build CSV row
            row = [stem, 0]
            for (x, y), c in zip(kpts15, scs15):
                row += [round(float(x), 3), round(float(y), 3), round(float(c), 4)]
            rows.append(row)

            # Render and save
            rendered = img.copy()
            if draw_skeleton is not None:
                try:
                    rendered = draw_skeleton(
                        rendered, kpts, scs,
                        openpose_skeleton=False, kpt_thr=KPT_THRESH,
                    )
                except Exception:
                    rendered = draw_15_skeleton(img.copy(), kpts15, scs15, KPT_THRESH)
            else:
                rendered = draw_15_skeleton(rendered, kpts15, scs15, KPT_THRESH)

            cv2.imwrite(str(out_dir / crop_path.name), rendered)

        # Write one CSV per frame
        csv_path = out_dir / f"{folder.name}_keypoints.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            w.writerows(rows)
        total_rows += len(rows)
        succeeded += 1
        print(f"        -> {len(rows)} row(s) in {csv_path.name}")

    print("\n-----------------------------------------")
    print("  RTMPOSE SUMMARY")
    print("-----------------------------------------")
    print(f"  Folders processed : {len(frame_folders)}")
    print(f"  Total crops fed   : {total_imgs}")
    print(f"  Total CSV rows    : {total_rows}")
    print("-----------------------------------------\n")


if __name__ == "__main__":
    main()
