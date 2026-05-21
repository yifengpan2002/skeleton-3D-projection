"""
main.py
--------
Full pipeline orchestrator: extract → tvcalib → yolo

Runs the 3 scripts in order, skipping steps that are already done.
"""

import os
import subprocess
import time
from pathlib import Path
import sys


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
VIDEO_PATH  = r"data\fifa\ARG_CRO_220001 (1).mp4"
FRAMES_ROOT = "data/frames"
OUTPUT_ROOT = "data/detections"
# ─────────────────────────────────────────


def main():
    video_name   = Path(VIDEO_PATH).stem
    frames_dir   = os.path.join(FRAMES_ROOT, video_name)
    output_dir   = os.path.join(OUTPUT_ROOT, video_name)
    npz_path     = os.path.join(output_dir, "homographies.npz")
    output_video = os.path.join(output_dir, f"{video_name}_annotated.mp4")
    csv_path     = os.path.join(output_dir, "detections.csv")

    print("=" * 50)
    print("  PLAYER DETECTION PIPELINE")
    print("=" * 50)
    print(f"  Video       : {VIDEO_PATH}")
    print(f"  Video name  : {video_name}")
    print(f"  Frames dir  : {frames_dir}")
    print(f"  Output dir  : {output_dir}")
    print(f"  NPZ file    : {npz_path}")
    print(f"  Output video: {output_video}")
    print("=" * 50 + "\n")

    total_start = time.time()

    # ═══════════════════════════════════════════════════════════════
    #  STEP 1: Extract Frames
    # ═══════════════════════════════════════════════════════════════
    print("=" * 50)
    print("  STEP 1 / 3  —  EXTRACT FRAMES")
    print("=" * 50 + "\n")

    if list(Path(frames_dir).glob("frame_*.jpg")):
        print(f"[Frames] Already exist in '{frames_dir}' — skipping.\n")
    else:
        print("[Frames] Running extract_frame.py...")
        subprocess.run([sys.executable, "extract_frame.py"], check=True)
        print()

    # ═══════════════════════════════════════════════════════════════
    #  STEP 2: TVCalib Homographies
    # ═══════════════════════════════════════════════════════════════
    print("=" * 50)
    print("  STEP 2 / 3  —  TVCALIB HOMOGRAPHIES")
    print("=" * 50 + "\n")

    if Path(npz_path).exists():
        print(f"[TVCalib] NPZ already exists at '{npz_path}' — skipping.\n")
    else:
        print("[TVCalib] Running tvcalib_preprocess.py...")
        subprocess.run([sys.executable, "tvcalib_preprocess.py"], check=True)
        print()

    # ═══════════════════════════════════════════════════════════════
    #  STEP 3: YOLO Detection + Filtering
    # ═══════════════════════════════════════════════════════════════
    print("=" * 50)
    print("  STEP 3 / 3  —  YOLO DETECTION + FIELD FILTERING")
    print("=" * 50 + "\n")

    print("[YOLO]   Running player_detection.py...")
    subprocess.run([sys.executable, "player_detection.py"], check=True)

    total_time = time.time() - total_start

    # ═══════════════════════════════════════════════════════════════
    #  FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 50)
    print("  PIPELINE COMPLETE")
    print("=" * 50)
    print(f"  Total time  : {total_time:.1f}s")
    print(f"  CSV         : {csv_path}")
    print(f"  Video       : {output_video}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()