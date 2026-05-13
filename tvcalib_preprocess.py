"""
tvcalib_preprocess.py
----------------------
Uses the REAL TVCalib API (from inference.ipynb) to generate homographies.

This will likely fail on rugby footage since the segmentation model
was trained on soccer fields. Run it to see what happens, then decide
whether to switch to manual calibration.
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path
from functools import partial
from multiprocessing import Pool
from collections import defaultdict

# ── TVCalib imports matching inference.ipynb ──────────────────────────────────
sys.path.insert(0, "tvcalib")
sys.path.insert(0, "tvcalib/sn_segmentation/src")

from tvcalib.module import TVCalibModule
from tvcalib.cam_distr.tv_main_center import get_cam_distr, get_dist_distr
from sn_segmentation.src.custom_extremities import generate_class_synthesis, get_line_extremities
from tvcalib.sncalib_dataset import custom_list_collate
from tvcalib.utils.io import detach_dict, tensor2list
from tvcalib.utils.objects_3d import SoccerPitchLineCircleSegments, SoccerPitchSNCircleCentralSplit
from tvcalib.inference import InferenceDatasetCalibration, InferenceDatasetSegmentation, InferenceSegmentationModel


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
VIDEO_NAME      = "Adam_Doueihi-Tigers_v_Eels_NRL_R6_2023"
FRAMES_ROOT     = "data/frames"
OUTPUT_ROOT     = "data/detections"
SEG_CHECKPOINT  = "tvcalib/data/segment_localization/train_59.pt"
IMAGE_WIDTH     = 1280
IMAGE_HEIGHT    = 720
BATCH_SIZE_SEG  = 16        # segmentation batch size
BATCH_SIZE_CALIB = 64       # calibration batch size (reduce if OOM)
OPTIM_STEPS     = 2000      # optimization iterations per batch
NWORKERS        = 4         # multiprocessing workers for point extraction
LENS_DIST       = False     # enable lens distortion modeling
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
# ─────────────────────────────────────────


def main():
    frames_dir = os.path.join(FRAMES_ROOT, VIDEO_NAME)
    output_dir = os.path.join(OUTPUT_ROOT, VIDEO_NAME)
    npz_path   = os.path.join(output_dir, "homographies.npz")

    print(f"[Setup]  Video name : {VIDEO_NAME}")
    print(f"[Setup]  Frames dir : {frames_dir}")
    print(f"[Setup]  NPZ output : {npz_path}")
    print(f"[Setup]  Device     : {DEVICE}\n")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 1: Segmentation → Extract keypoints from frames
    # ═══════════════════════════════════════════════════════════════
    print("[1/3]    Running field line segmentation...")

    dataset_seg = InferenceDatasetSegmentation(
        Path(frames_dir), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    dataloader_seg = torch.utils.data.DataLoader(
        dataset_seg,
        batch_size=BATCH_SIZE_SEG,
        num_workers=0,  # Windows multiprocessing issues; set to NWORKERS on Linux
        shuffle=False,
        collate_fn=custom_list_collate,
    )

    model_seg = InferenceSegmentationModel(SEG_CHECKPOINT, DEVICE)

    # Prepare point extraction functions
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

    for batch_dict in dataloader_seg:
        # Semantic segmentation
        with torch.no_grad():
            sem_lines = model_seg.inference(batch_dict["image"].to(DEVICE))
        sem_lines = sem_lines.cpu().numpy().astype(np.uint8)

        # Extract line extremities as keypoints
        with Pool(NWORKERS) as p:
            skeletons_batch = p.map(fn_generate_class_synthesis, sem_lines)
            keypoints_raw_batch = p.map(fn_get_line_extremities, skeletons_batch)

        image_ids.extend(batch_dict["image_id"])
        keypoints_raw.extend(keypoints_raw_batch)

    print(f"[1/3]    Extracted keypoints from {len(image_ids)} frames\n")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 2: Camera calibration via TVCalib optimization
    # ═══════════════════════════════════════════════════════════════
    print("[2/3]    Running TVCalib camera calibration...")

    object3d = SoccerPitchLineCircleSegments(
        device=DEVICE, base_field=SoccerPitchSNCircleCentralSplit()
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
        keypoints_raw, IMAGE_WIDTH, IMAGE_HEIGHT, object3d
    )
    dataloader_calib = torch.utils.data.DataLoader(
        dataset_calib, BATCH_SIZE_CALIB, collate_fn=custom_list_collate
    )

    per_sample_output = defaultdict(list)
    per_sample_output["image_id"] = [[x] for x in image_ids]

    for x_dict in dataloader_calib:
        _batch_size = x_dict["lines__ndc_projected_selection_shuffled"].shape[0]

        per_sample_loss, cam, _ = model_calib.self_optim_batch(x_dict)
        output_dict = tensor2list(
            detach_dict({**cam.get_parameters(_batch_size), **per_sample_loss})
        )

        for k in output_dict.keys():
            per_sample_output[k].extend(output_dict[k])

    print(f"[2/3]    Calibrated {len(image_ids)} frames\n")

    # ═══════════════════════════════════════════════════════════════
    #  STEP 3: Extract homographies and save to NPZ
    # ═══════════════════════════════════════════════════════════════
    print("[3/3]    Extracting homographies...")

    homographies = per_sample_output["homography"]  # List of (1, 3, 3) arrays
    frame_names = [Path(img_id[0]).stem for img_id in per_sample_output["image_id"]]

    # Convert to (N, 3, 3) array
    H_array = np.stack([np.array(h).squeeze() for h in homographies], axis=0)
    names_array = np.array(frame_names)

    # Check for failed calibrations (NaN in homography)
    valid_mask = ~np.isnan(H_array).any(axis=(1, 2))
    valid_count = valid_mask.sum()
    failed_count = len(H_array) - valid_count

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, frame_names=names_array, homographies=H_array)

    print(f"[3/3]    Saved {len(frame_names)} homographies ({valid_count} valid, "
          f"{failed_count} failed) → '{npz_path}'")
    print(f"         Keys: 'frame_names' {names_array.shape}, "
          f"'homographies' {H_array.shape} float64\n")

    if failed_count > 0:
        print(f"[Warning] {failed_count} frames failed calibration (NaN homographies).")
        print(f"          This is expected for rugby footage since the model was")
        print(f"          trained on soccer fields with different line markings.\n")

    print("[Done]   TVCalib preprocessing complete.")
    print(f"         Load in player_detection.py with:")
    print(f"         >>> data = np.load('{npz_path}')")
    print(f"         >>> H = data['homographies'][i]")


if __name__ == "__main__":
    main()