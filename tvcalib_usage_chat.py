import sys
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from collections import defaultdict

# ============================================================
# Project path setup
# Run this file from:
# C:\Users\Yifeng Pan\Documents\compsci760\skeleton-3D-projection
# ============================================================

ROOT = Path(__file__).resolve().parent

sys.path.insert(0, str(ROOT / "tvcalib"))
sys.path.insert(0, str(ROOT / "tvcalib" / "sn_segmentation" / "src"))

from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import (
    generate_class_synthesis,
    get_line_extremities,
)
from tvcalib.sncalib_dataset import custom_list_collate
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import (
    SoccerPitchLineCircleSegments,
    SoccerPitchSNCircleCentralSplit,
)
from tvcalib.inference import (
    InferenceDatasetCalibration,
    InferenceDatasetSegmentation,
    InferenceSegmentationModel,
)

# ============================================================
# Configuration
# ============================================================

VIDEO_NAME = "ARG_CRO_220001 (1)"
FRAMES_ROOT = ROOT / "data" / "single_football_frame"
OUTPUT_ROOT = ROOT / "data" / "detections"

SEG_CHECKPOINT = ROOT / "tvcalib" / "data" / "segment_localization" / "train_59.pt"

IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720

NUM_FRAMES = 2         # test first 10 frames first
BATCH_SIZE_SEG = 1
BATCH_SIZE_CALIB = 1     # keep small on CPU
OPTIM_STEPS = 2000
NWORKERS = 0             # safer on Windows
LENS_DIST = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ============================================================
# Soccer pitch template for visualization
# 105m x 68m, SoccerNet-style field
# ============================================================

SOCCER_FIELD_LINES = {
    # Goal lines
    "left_goal_line":    [(-52.5, -34), (-52.5, 34)],
    "right_goal_line":   [(52.5, -34), (52.5, 34)],

    # Sidelines
    "top_sideline":      [(-52.5, -34), (52.5, -34)],
    "bottom_sideline":   [(-52.5, 34), (52.5, 34)],

    # Halfway line
    "halfway":           [(0, -34), (0, 34)],

    # Left penalty box (16.5m from goal, 20.16m each side of center)
    "left_box_top":      [(-52.5, -20.16), (-36, -20.16)],
    "left_box_bottom":   [(-52.5, 20.16), (-36, 20.16)],
    "left_box_right":    [(-36, -20.16), (-36, 20.16)],

    # Right penalty box
    "right_box_top":     [(52.5, -20.16), (36, -20.16)],
    "right_box_bottom":  [(52.5, 20.16), (36, 20.16)],
    "right_box_left":    [(36, -20.16), (36, 20.16)],
}


LINE_COLORS = {
    0:  (0, 0, 0),
    1:  (255, 0, 0),
    2:  (0, 255, 0),
    3:  (0, 0, 255),
    4:  (255, 255, 0),
    5:  (255, 0, 255),
    6:  (0, 255, 255),
    7:  (128, 0, 0),
    8:  (0, 128, 0),
    9:  (0, 0, 128),
    10: (128, 128, 0),
    11: (128, 0, 128),
    12: (0, 128, 128),
    13: (192, 192, 192),
}


def colorize_segmentation(seg_mask):
    h, w = seg_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)

    for class_id, color in LINE_COLORS.items():
        colored[seg_mask == class_id] = color

    return colored


def save_segmentation_overlay(frame, seg_mask, out_path):
    seg_colored = colorize_segmentation(seg_mask)
    seg_resized = cv2.resize(
        seg_colored,
        (frame.shape[1], frame.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    )

    overlay = cv2.addWeighted(frame, 0.45, seg_resized, 0.55, 0)
    cv2.imwrite(str(out_path), overlay)


def project_points(points_world, H, use_inverse=True):
    """
    Try projecting world coordinates to image coordinates.

    TVCalib homography direction can be confusing depending on API output.
    Therefore this script saves two visualizations:
    1. using inverse(H)
    2. using H directly
    """
    if use_inverse:
        M = np.linalg.inv(H)
    else:
        M = H

    world_homo = np.concatenate(
        [points_world, np.ones((len(points_world), 1), dtype=np.float32)],
        axis=1,
    )

    pix_homo = (M @ world_homo.T).T
    pix = pix_homo[:, :2] / pix_homo[:, 2:3]

    return pix.astype(np.int32)


def draw_full_pitch(frame, H, out_path, use_inverse=True):
    vis = frame.copy()
    h, w = frame.shape[:2]

    # Draw straight field lines
    for name, (p1, p2) in SOCCER_FIELD_LINES.items():
        if "center_circle" in name:
            continue

        world_pts = np.array([p1, p2], dtype=np.float32)

        try:
            pixel_pts = project_points(world_pts, H, use_inverse=use_inverse)
        except Exception:
            continue

        pt1 = tuple(pixel_pts[0])
        pt2 = tuple(pixel_pts[1])

        margin = 300
        if not (
            -margin < pt1[0] < w + margin and
            -margin < pt1[1] < h + margin and
            -margin < pt2[0] < w + margin and
            -margin < pt2[1] < h + margin
        ):
            continue

        if "goal_line" in name:
            color = (0, 0, 255)
            thickness = 3
        elif "sideline" in name:
            color = (255, 0, 0)
            thickness = 3
        elif "halfway" in name:
            color = (0, 255, 0)
            thickness = 3
        else:
            color = (0, 255, 255)
            thickness = 2

        cv2.line(vis, pt1, pt2, color, thickness)

    # Draw center circle by sampling points
    circle_world = []
    cx, cy = 0, 0   # Center of field (was 52.5, 34)
    r = 9.15

    for t in np.linspace(0, 2 * np.pi, 120):
        circle_world.append([cx + r * np.cos(t), cy + r * np.sin(t)])

    circle_world = np.array(circle_world, dtype=np.float32)

    try:
        circle_px = project_points(circle_world, H, use_inverse=use_inverse)
        cv2.polylines(vis, [circle_px], isClosed=True, color=(255, 0, 255), thickness=2)
    except Exception:
        pass

    label = "Projection: inverse(H)" if use_inverse else "Projection: H directly"
    cv2.putText(
        vis,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        3,
    )
    cv2.putText(
        vis,
        label,
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 0, 0),
        1,
    )

    cv2.imwrite(str(out_path), vis)


def main():
    frames_dir = FRAMES_ROOT 
    output_dir = OUTPUT_ROOT / VIDEO_NAME / "tvcalib_correct_debug"
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_path = OUTPUT_ROOT / VIDEO_NAME / "homographies_tvcalib_correct.npz"

    print("=" * 70)
    print(" TVCalib Correct API Run")
    print("=" * 70)
    print(f"Frames dir : {frames_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Checkpoint : {SEG_CHECKPOINT}")
    print(f"Device     : {DEVICE}")
    print("=" * 70)

    if not frames_dir.exists():
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    frame_files = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No frame_*.jpg found in: {frames_dir}")

    frame_files = frame_files[:NUM_FRAMES]

    print(f"\n[Frames] Using {len(frame_files)} frame(s)")

    # ============================================================
    # Step 1: TVCalib segmentation
    # ============================================================

    print("\n[1/4] Loading segmentation dataset/model...")

    dataset_seg = InferenceDatasetSegmentation(
        Path(frames_dir),
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
    )

    dataset_seg = torch.utils.data.Subset(dataset_seg, range(len(frame_files)))

    dataloader_seg = torch.utils.data.DataLoader(
        dataset_seg,
        batch_size=BATCH_SIZE_SEG,
        num_workers=0,
        shuffle=False,
        collate_fn=custom_list_collate,
    )

    model_seg = InferenceSegmentationModel(str(SEG_CHECKPOINT), DEVICE)

    fn_generate_class_synthesis = partial(generate_class_synthesis, radius=4)
    fn_get_line_extremities = partial(
        get_line_extremities,
        maxdist=30,
        width=455,
        height=256,
        num_points_lines=4,
        num_points_circles=8,
    )

    image_ids = []
    keypoints_raw = []
    segmentation_masks = {}

    print("[1/4] Running segmentation...")

    for batch_dict in dataloader_seg:
        with torch.no_grad():
            sem_lines = model_seg.inference(batch_dict["image"].to(DEVICE))

        sem_lines = sem_lines.cpu().numpy().astype(np.uint8)

        if NWORKERS > 0:
            with Pool(NWORKERS) as p:
                skeletons_batch = p.map(fn_generate_class_synthesis, sem_lines)
                keypoints_raw_batch = p.map(fn_get_line_extremities, skeletons_batch)
        else:
            skeletons_batch = [fn_generate_class_synthesis(x) for x in sem_lines]
            keypoints_raw_batch = [fn_get_line_extremities(x) for x in skeletons_batch]

        for image_id, seg_mask in zip(batch_dict["image_id"], sem_lines):
            segmentation_masks[str(image_id)] = seg_mask

        image_ids.extend(batch_dict["image_id"])
        keypoints_raw.extend(keypoints_raw_batch)

    print(f"[1/4] Extracted keypoints from {len(image_ids)} frames")

    # Save segmentation overlays
    seg_dir = output_dir / "01_segmentation_overlay"
    seg_dir.mkdir(parents=True, exist_ok=True)

    for image_id in image_ids:
        frame_path = frames_dir / image_id
        frame = cv2.imread(str(frame_path))
        if frame is None:
            continue

        out_path = seg_dir / f"{Path(image_id).stem}_seg_overlay.jpg"
        save_segmentation_overlay(frame, segmentation_masks[str(image_id)], out_path)

    print(f"[1/4] Saved segmentation overlays to: {seg_dir}")

    # ============================================================
    # Step 2: TVCalib calibration
    # ============================================================

    print("\n[2/4] Building soccer pitch calibration model...")

    object3d = SoccerPitchLineCircleSegments(
        device=DEVICE,
        base_field=SoccerPitchSNCircleCentralSplit(),
    )

    model_calib = TVCalibModule(
        object3d,
        get_cam_distr(1.96, BATCH_SIZE_CALIB, 1),
        get_dist_distr(BATCH_SIZE_CALIB, 1) if LENS_DIST else None,
        (IMAGE_HEIGHT, IMAGE_WIDTH),
        OPTIM_STEPS,
        DEVICE,
        log_per_step=False,
        tqdm_kwqargs={"desc": "Optimizing", "leave": False},
    )

    dataset_calib = InferenceDatasetCalibration(
        keypoints_raw,
        IMAGE_WIDTH,
        IMAGE_HEIGHT,
        object3d,
    )

    dataloader_calib = torch.utils.data.DataLoader(
        dataset_calib,
        batch_size=BATCH_SIZE_CALIB,
        collate_fn=custom_list_collate,
    )

    per_sample_output = defaultdict(list)
    per_sample_output["image_id"] = [[x] for x in image_ids]

    print("[2/4] Running camera self-optimization...")

    for x_dict in dataloader_calib:
        batch_size = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]

        per_sample_loss, cam, _ = model_calib.self_optim_batch(x_dict)

        output_dict = tensor2list(
            detach_dict({
                **cam.get_parameters(batch_size),
                **per_sample_loss,
            })
        )

        for k, v in output_dict.items():
            per_sample_output[k].extend(v)

    # ============================================================
    # Step 3: Save homographies
    # ============================================================

    print("\n[3/4] Saving homographies...")

    homographies = per_sample_output["homography"]
    frame_names = [Path(img_id[0]).stem for img_id in per_sample_output["image_id"]]

    H_array = np.stack([np.array(h).squeeze() for h in homographies], axis=0)
    names_array = np.array(frame_names)

    valid_mask = ~np.isnan(H_array).any(axis=(1, 2))
    valid_count = int(valid_mask.sum())
    failed_count = len(H_array) - valid_count

    np.savez_compressed(
        npz_path,
        frame_names=names_array,
        homographies=H_array,
    )

    print(f"[3/4] Saved: {npz_path}")
    print(f"[3/4] Valid: {valid_count}, Failed: {failed_count}")

    # ============================================================
    # Step 4: Draw full pitch projection
    # ============================================================

    print("\n[4/4] Drawing full pitch projections...")

    vis_inv_dir = output_dir / "02_full_pitch_inverse_H"
    vis_raw_dir = output_dir / "03_full_pitch_raw_H"

    vis_inv_dir.mkdir(parents=True, exist_ok=True)
    vis_raw_dir.mkdir(parents=True, exist_ok=True)

    for frame_name, H in zip(frame_names, H_array):
        if np.isnan(H).any():
            print(f"  {frame_name}: failed homography, skipping")
            continue

        frame_path = frames_dir / f"{frame_name}.jpg"
        frame = cv2.imread(str(frame_path))

        if frame is None:
            print(f"  {frame_name}: frame missing, skipping")
            continue

        draw_full_pitch(
            frame,
            H,
            vis_inv_dir / f"{frame_name}_inverse_H.jpg",
            use_inverse=True,
        )

        draw_full_pitch(
            frame,
            H,
            vis_raw_dir / f"{frame_name}_raw_H.jpg",
            use_inverse=False,
        )

        print(f"  {frame_name}: saved projection images")

    print("\n[Done]")
    print(f"Check these folders:")
    print(f"  Segmentation overlay : {seg_dir}")
    print(f"  Full pitch inverse H : {vis_inv_dir}")
    print(f"  Full pitch raw H     : {vis_raw_dir}")
    print(f"  Homographies NPZ     : {npz_path}")
    print("\nLook at both projection folders.")
    print("Whichever one aligns with the field tells you the correct H direction.")

    # After getting H_array, test what coordinates H produces
    H = H_array[0]  # First frame's homography

    # Pick a known pixel (center of the image)
    h, w = IMAGE_HEIGHT, IMAGE_WIDTH
    test_pixels = np.array([
        [w/2, h/2],      # Image center
        [0, 0],          # Top-left
        [w, 0],          # Top-right
        [0, h],          # Bottom-left
        [w, h],          # Bottom-right
    ], dtype=np.float32)

    # Project through H
    for px in test_pixels:
        pt = np.array([px[0], px[1], 1.0])
        world = H @ pt
        world = world[:2] / world[2]
        print(f"  Pixel ({px[0]:.0f}, {px[1]:.0f}) -> World ({world[0]:.2f}, {world[1]:.2f})")


if __name__ == "__main__":
    main()