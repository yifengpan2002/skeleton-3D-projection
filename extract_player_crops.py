"""
extract_player_crops.py
------------------------
Reads bounding boxes from detections.csv and crops each detected
player out of its source frame. Crops are saved into a subfolder
named after the frame, inside the same output folder.

Output structure:
    outputs/player_detect/annotated_player/
        ├── detections.csv
        ├── frame_000000.jpg              ← annotated frame (from detection step)
        ├── frame_000000/                 ← NEW: subfolder of crops
        │   ├── player_0.jpg
        │   ├── player_1.jpg
        │   └── ...
        ├── frame_000020/
        │   └── ...
        └── ...
"""

import cv2
import csv
from pathlib import Path


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
CSV_PATH    = "outputs/player_detect/annotated_player/detections.csv"
OUTPUT_DIR  = "outputs/player_detect/annotated_player"
PADDING     = 10      # optional pixels of padding around each crop
JPG_QUALITY = 95
# ─────────────────────────────────────────


def crop_players_from_csv(
    csv_path: str,
    output_dir: str,
    padding: int = 0,
    jpg_quality: int = 95,
) -> None:
    """
    Read CSV of detections and crop each bounding box out of its
    source frame. Crops are saved into <output_dir>/<frame_stem>/.
    """
    csv_file   = Path(csv_path)
    output_dir = Path(output_dir)

    if not csv_file.exists():
        raise FileNotFoundError(f"CSV not found: '{csv_path}'")

    # Group rows by frame so we open each frame only once
    rows_by_frame: dict[str, list[dict]] = {}
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows_by_frame.setdefault(row["frame_path"], []).append(row)

    print(f"[CSV]    Loaded {sum(len(v) for v in rows_by_frame.values())} "
          f"detections across {len(rows_by_frame)} frames")
    print(f"[Setup]  Output dir : {output_dir}")
    print(f"[Setup]  Padding    : {padding} px\n")

    total_saved   = 0
    total_skipped = 0

    for frame_path, rows in rows_by_frame.items():
        frame_file = Path(frame_path)
        frame_stem = frame_file.stem            # e.g. "frame_000000"

        img = cv2.imread(frame_path)
        if img is None:
            print(f"  [skip] Could not read '{frame_path}'")
            total_skipped += len(rows)
            continue

        h, w = img.shape[:2]
        crop_dir = output_dir / frame_stem
        crop_dir.mkdir(parents=True, exist_ok=True)

        for row in rows:
            player_id = int(row["player_id"])
            x1 = int(row["x1"]); y1 = int(row["y1"])
            x2 = int(row["x2"]); y2 = int(row["y2"])

            # Apply optional padding and clamp to image bounds
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            if x2 <= x1 or y2 <= y1:
                total_skipped += 1
                continue

            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                total_skipped += 1
                continue

            out_path = crop_dir / f"player_{player_id}.jpg"
            cv2.imwrite(
                str(out_path),
                crop,
                [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality],
            )
            total_saved += 1

        print(f"  {frame_stem} -> {len(rows)} crop(s) "
              f"in '{crop_dir.relative_to(output_dir.parent)}'")

    print("\n-----------------------------------------")
    print("  EXTRACTION SUMMARY")
    print("-----------------------------------------")
    print(f"  Frames processed : {len(rows_by_frame)}")
    print(f"  Crops saved      : {total_saved}")
    print(f"  Crops skipped    : {total_skipped}")
    print("-----------------------------------------\n")


# ─────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    crop_players_from_csv(
        csv_path    = CSV_PATH,
        output_dir  = OUTPUT_DIR,
        padding     = PADDING,
        jpg_quality = JPG_QUALITY,
    )
