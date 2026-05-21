"""
extract_frame.py
-----------------
Splits a video file into individual JPEG frames.
Frames are saved under:  <frames_root>/<video_name>/frame_XXXXXX.jpg

Can be run standalone or imported by main.py.
"""

import cv2
import os
from pathlib import Path


def extract_frames(
    video_path: str,
    frames_root: str = "data/frames",
    frame_step: int = 1,
) -> tuple[list[str], str]:
    """
    Extract frames from a video and save as JPEG images.

    Args:
        video_path  : path to the input video file
        frames_root : root directory for frame output
        frame_step  : extract every N-th frame (1 = every frame)

    Returns:
        frame_paths : sorted list of saved frame file paths
        frames_dir  : directory where frames were saved
    """
    video_name = Path(video_path).stem
    frames_dir = os.path.join(frames_root, video_name)
    Path(frames_dir).mkdir(parents=True, exist_ok=True)

    # Skip if frames already exist
    existing = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if existing:
        print(f"[Frames] Found {len(existing)} existing frames in '{frames_dir}' — skipping extraction.")
        return [str(p) for p in existing], frames_dir

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[Frames] Video    : {video_path}")
    print(f"         Info     : {total_frames} frames | {fps:.2f} fps | {width}x{height}")
    print(f"         Step     : every {frame_step} frame(s)")
    print(f"         Output   : {frames_dir}")

    saved_paths = []
    frame_idx   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            filename = os.path.join(frames_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_paths.append(filename)
        frame_idx += 1

    cap.release()
    print(f"[Frames] Saved {len(saved_paths)} frames -> '{frames_dir}'\n")
    return sorted(saved_paths), frames_dir


# ─────────────────────────────────────────
#  STANDALONE ENTRY POINT
# ─────────────────────────────────────────
if __name__ == "__main__":
    VIDEO_PATH  = "data/fifa/ARG_CRO_220001 (1).mp4"
    FRAMES_ROOT = "data/frames"
    FRAME_STEP  = 1

    paths, out_dir = extract_frames(VIDEO_PATH, FRAMES_ROOT, FRAME_STEP)
    print(f"[Done]   {len(paths)} frames ready in '{out_dir}'")