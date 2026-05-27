"""
run_openpose_on_crops.py
-------------------------
Runs OpenPose on each cropped player image produced by
extract_player_crops.py. Each crop contains a single player, which
gives OpenPose cleaner, more accurate keypoints than running it on
the whole frame.

Expected input structure:
    outputs/player_detect/annotated_player/
        ├── frame_000000/
        │   ├── player_0.jpg
        │   └── player_1.jpg
        ├── frame_000020/
        │   └── ...

Output structure (created next to the crops folders):
    outputs/player_detect/annotated_player/
        ├── frame_000000/                       (crops)
        ├── frame_000000_openpose/              ← NEW
        │   ├── player_0.jpg                    (rendered skeleton)
        │   ├── player_1.jpg
        │   ├── player_0_keypoints.json         (raw OpenPose output)
        │   ├── player_1_keypoints.json
        │   └── frame_000000_keypoints.csv      (combined CSV for this frame)
        ├── frame_000020_openpose/
        │   └── ...
"""

import subprocess
import sys
import json
import csv
from pathlib import Path


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
# Path to your OpenPose installation (folder containing bin/, models/, etc.)
OPENPOSE_DIR    = r"C:\Users\Yifeng Pan\Documents\compsci760\openpose"

# Same folder that holds the crops (frame_XXXXXX/ subfolders)
ROOT_DIR        = "outputs/player_detect/annotated_player"

# OpenPose runtime options
NET_RESOLUTION  = "-1x368"   # default; use "-1x256" for more speed on small crops
MODEL_POSE      = "BODY_25"  # BODY_25 | COCO | MPI
USE_GPU         = True
HAND            = False
FACE            = False

# Cleanup: remove per-image JSON files once they've been merged into the CSV
DELETE_JSON_AFTER_CSV = False
# ─────────────────────────────────────────


# ─────────────────────────────────────────
#  KEYPOINT NAMES
# ─────────────────────────────────────────
KEYPOINT_NAMES = {
    "BODY_25": [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
        "LShoulder", "LElbow", "LWrist", "MidHip", "RHip",
        "RKnee", "RAnkle", "LHip", "LKnee", "LAnkle",
        "REye", "LEye", "REar", "LEar",
        "LBigToe", "LSmallToe", "LHeel",
        "RBigToe", "RSmallToe", "RHeel",
    ],
    "COCO": [
        "Nose", "Neck", "RShoulder", "RElbow", "RWrist",
        "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
        "RAnkle", "LHip", "LKnee", "LAnkle",
        "REye", "LEye", "REar", "LEar",
    ],
    "MPI": [
        "Head", "Neck", "RShoulder", "RElbow", "RWrist",
        "LShoulder", "LElbow", "LWrist", "RHip", "RKnee",
        "RAnkle", "LHip", "LKnee", "LAnkle", "Chest",
    ],
}


# ─────────────────────────────────────────
#  OPENPOSE RUNNER
# ─────────────────────────────────────────

def run_openpose_on_folder(
    input_dir: Path,
    output_dir: Path,
    openpose_dir: Path,
) -> bool:
    """Run OpenPose on every image in input_dir. Outputs go to output_dir."""
    openpose_bin = openpose_dir / "bin" / "OpenPoseDemo.exe"
    if not openpose_bin.exists():
        raise FileNotFoundError(f"OpenPose binary not found: {openpose_bin}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(openpose_bin),
        "--image_dir",      str(input_dir.resolve()),
        "--write_json",     str(output_dir.resolve()),
        "--write_images",   str(output_dir.resolve()),
        "--model_pose",     MODEL_POSE,
        "--net_resolution", NET_RESOLUTION,
        "--display",        "0",
    ]
    if not USE_GPU:
        cmd += ["--num_gpu", "0", "--num_gpu_start", "0"]
    if HAND:
        cmd += ["--hand"]
    if FACE:
        cmd += ["--face"]

    try:
        # cwd must be the OpenPose directory so its DLLs and models resolve
        subprocess.run(
            cmd,
            cwd=str(openpose_dir),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"  [error] OpenPose failed on '{input_dir.name}'")
        if e.stderr:
            print(e.stderr.decode(errors="ignore"))
        return False


# ─────────────────────────────────────────
#  JSON -> CSV
# ─────────────────────────────────────────

def merge_jsons_to_csv(
    openpose_out_dir: Path,
    csv_path: Path,
    model_pose: str,
) -> int:
    """
    Read all *_keypoints.json files in openpose_out_dir and produce a
    single CSV. One row per detected person (typically one per crop).
    Returns the number of rows written.
    """
    keypoint_names = KEYPOINT_NAMES[model_pose]

    # Header: player_id, then x_y_c per keypoint
    header = ["player_id", "person_index"]
    for kp in keypoint_names:
        header += [f"{kp}_x", f"{kp}_y", f"{kp}_c"]

    rows = []
    json_files = sorted(openpose_out_dir.glob("*_keypoints.json"))
    for jf in json_files:
        # filename pattern: player_0_keypoints.json -> player_id = "player_0"
        stem = jf.stem.replace("_keypoints", "")

        with open(jf, "r") as f:
            data = json.load(f)

        people = data.get("people", [])
        if not people:
            # No person detected — still write an empty row so it's traceable
            row = [stem, -1] + [""] * (len(keypoint_names) * 3)
            rows.append(row)
            continue

        for person_idx, person in enumerate(people):
            kps = person.get("pose_keypoints_2d", [])
            expected_len = len(keypoint_names) * 3
            if len(kps) < expected_len:
                kps = kps + [0.0] * (expected_len - len(kps))
            else:
                kps = kps[:expected_len]
            row = [stem, person_idx] + [round(float(v), 3) for v in kps]
            rows.append(row)

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    if DELETE_JSON_AFTER_CSV:
        for jf in json_files:
            jf.unlink()

    return len(rows)


# ─────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────

def main() -> None:
    openpose_dir = Path(OPENPOSE_DIR)
    root_dir     = Path(ROOT_DIR)

    if not openpose_dir.exists():
        sys.exit(f"[error] OpenPose dir not found: {openpose_dir}")
    if not root_dir.exists():
        sys.exit(f"[error] Root dir not found: {root_dir}")
    if MODEL_POSE not in KEYPOINT_NAMES:
        sys.exit(f"[error] Unknown MODEL_POSE: {MODEL_POSE}")

    # Find frame subfolders containing crops; skip _openpose folders
    frame_folders = sorted(
        p for p in root_dir.iterdir()
        if p.is_dir()
        and not p.name.endswith("_openpose")
        and any(p.glob("*.jpg"))
    )

    if not frame_folders:
        sys.exit(f"[error] No frame_* crop subfolders found in '{root_dir}'")

    print(f"[Setup]  OpenPose dir : {openpose_dir}")
    print(f"[Setup]  Root dir     : {root_dir}")
    print(f"[Setup]  Net res      : {NET_RESOLUTION}")
    print(f"[Setup]  Model pose   : {MODEL_POSE}")
    print(f"[Setup]  Use GPU      : {USE_GPU}")
    print(f"[Setup]  Crop folders : {len(frame_folders)}\n")

    succeeded      = 0
    failed         = 0
    total_imgs     = 0
    total_csv_rows = 0

    for i, folder in enumerate(frame_folders, start=1):
        out_dir = root_dir / f"{folder.name}_openpose"
        n_imgs  = len(list(folder.glob("*.jpg")))
        total_imgs += n_imgs

        print(f"  [{i}/{len(frame_folders)}] {folder.name} "
              f"({n_imgs} crop(s)) -> '{out_dir.name}'")

        ok = run_openpose_on_folder(folder, out_dir, openpose_dir)
        if not ok:
            failed += 1
            continue

        # Merge per-crop JSONs into one CSV for this frame
        csv_path = out_dir / f"{folder.name}_keypoints.csv"
        n_rows = merge_jsons_to_csv(out_dir, csv_path, MODEL_POSE)
        total_csv_rows += n_rows
        succeeded += 1
        print(f"        -> {n_rows} row(s) in {csv_path.name}")

    print("\n-----------------------------------------")
    print("  OPENPOSE SUMMARY")
    print("-----------------------------------------")
    print(f"  Folders processed  : {len(frame_folders)}")
    print(f"  Succeeded          : {succeeded}")
    print(f"  Failed             : {failed}")
    print(f"  Total crops fed    : {total_imgs}")
    print(f"  Total CSV rows     : {total_csv_rows}")
    print("-----------------------------------------\n")


if __name__ == "__main__":
    main()
