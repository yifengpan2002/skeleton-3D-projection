"""
run_hrnet_on_crops.py
----------------------
Runs HRNet-w32 (COCO-Wholebody, 133 keypoints) on each cropped player
image produced by extract_player_crops.py. Each crop holds a single
player, so we pass the full crop area as the bounding box.

Keypoints are mapped from COCO-Wholebody (133) to the 15-joint format
used in the worldpose evaluation pipeline (includes toes via the
wholebody foot keypoints).

The HRNet model config is built programmatically (same approach as
worldpose_hrnet_wholebody_eval.py) to bypass Windows long-path issues
with MMPose's bundled config files.

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
        ├── frame_000000_hrnet/                 ← NEW
        │   ├── player_0.jpg                    (rendered skeleton)
        │   ├── player_1.jpg
        │   └── frame_000000_keypoints.csv      (one CSV per frame)
        ├── frame_000020_hrnet/
        │   └── ...

Dependencies
------------
  pip install mmpose mmengine mmcv (or mmcv-lite)
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
ROOT_DIR    = "outputs/player_detect/annotated_player"

HRNET_TAG   = "hrnet_w32_256x192"   # "hrnet_w32_256x192" or "hrnet_w32_384x288"
KPT_THRESH  = 0.3                   # confidence threshold for drawing keypoints
# ─────────────────────────────────────────


# ─────────────────────────────────────────
#  Joint mapping (from worldpose_hrnet_wholebody_eval.py)
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
    20,   # 13: Rfoot
    17,   # 14: Lfoot
]

JOINT_NAMES = [
    "nose", "Rshoulder", "Lshoulder", "Relbow", "Lelbow",
    "Rwrist", "Lwrist", "Rhip", "Lhip", "Rknee",
    "Lknee", "Rankle", "Lankle", "Rfoot", "Lfoot",
]

SKELETON_15 = [
    (0, 1), (0, 2),
    (1, 3), (3, 5),
    (2, 4), (4, 6),
    (1, 7), (2, 8), (7, 8),
    (7, 9), (9, 11), (11, 13),
    (8, 10), (10, 12), (12, 14),
]


# ─────────────────────────────────────────
#  HRNet model configurations
# ─────────────────────────────────────────
HRNET_MODELS = {
    'hrnet_w32_256x192': {
        'input_size':   (192, 256),
        'heatmap_size': (48, 64),
        'checkpoint':   'https://download.openmmlab.com/mmpose/top_down/hrnet/'
                        'hrnet_w32_coco_wholebody_256x192-853765cd_20200918.pth',
    },
    'hrnet_w32_384x288': {
        'input_size':   (288, 384),
        'heatmap_size': (72, 96),
        'checkpoint':   'https://download.openmmlab.com/mmpose/top_down/hrnet/'
                        'hrnet_w32_coco_wholebody_384x288-78cacac3_20200922.pth',
    },
}


# ─────────────────────────────────────────
#  COCO-Wholebody 133 dataset metadata
# ─────────────────────────────────────────
def _build_wholebody_flip_indices():
    body = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    foot = [20, 21, 22, 17, 18, 19]
    face_local = [
        16,15,14,13,12,11,10,9,8,7,6,5,4,3,2,1,0,
        26,25,24,23,22,21,20,19,18,17,
        27,28,29,30,35,34,33,32,31,
        45,44,43,42,47,46,39,38,37,36,41,40,
        54,53,52,51,50,49,48,59,58,57,56,55,
        64,63,62,61,60,67,66,65,
    ]
    face = [i + 23 for i in face_local]
    lhand = list(range(112, 133))
    rhand = list(range(91, 112))
    return body + foot + face + lhand + rhand

WHOLEBODY133_FLIP_INDICES = _build_wholebody_flip_indices()

_BODY_SIGMAS = [0.026,0.025,0.025,0.035,0.035,0.079,0.079,0.072,0.072,
                0.062,0.062,0.107,0.107,0.087,0.087,0.089,0.089]
_FOOT_SIGMAS = [0.068,0.066,0.066,0.092,0.094,0.094]
_FACE_SIGMAS = [
    0.042,0.043,0.044,0.043,0.040,0.035,0.031,0.025,0.020,0.023,0.029,
    0.032,0.037,0.038,0.043,0.041,0.045,0.013,0.012,0.011,0.011,0.012,
    0.012,0.011,0.011,0.013,0.015,0.009,0.007,0.007,0.007,0.012,0.009,
    0.008,0.016,0.010,0.017,0.011,0.009,0.011,0.009,0.007,0.013,0.008,
    0.011,0.012,0.010,0.034,0.008,0.008,0.009,0.008,0.008,0.007,0.010,
    0.008,0.009,0.009,0.009,0.007,0.007,0.008,0.011,0.008,0.008,0.008,
    0.01,0.008]
_HAND_SIGMAS = [
    0.029,0.022,0.035,0.037,0.047,0.026,0.025,0.024,0.035,0.018,0.024,
    0.022,0.026,0.017,0.021,0.021,0.032,0.020,0.019,0.022,0.031] * 2

COCO_WHOLEBODY_DATASET_META = dict(
    dataset_name='coco_wholebody',
    num_keypoints=133,
    keypoint_id2name={i: f'kp_{i}' for i in range(133)},
    keypoint_name2id={f'kp_{i}': i for i in range(133)},
    upper_body_ids=[0,1,2,3,4,5,6,7,8,9,10] + list(range(23, 133)),
    lower_body_ids=[11,12,13,14,15,16,17,18,19,20,21,22],
    flip_indices=WHOLEBODY133_FLIP_INDICES,
    flip_pairs=[[i, j] for i, j in enumerate(WHOLEBODY133_FLIP_INDICES) if i < j],
    skeleton_links=[],
    sigmas=_BODY_SIGMAS + _FOOT_SIGMAS + _FACE_SIGMAS + _HAND_SIGMAS,
)


# ─────────────────────────────────────────
#  Build HRNet config programmatically
# ─────────────────────────────────────────
def build_hrnet_config(input_size, heatmap_size):
    from mmengine.config import Config

    codec = dict(
        type='MSRAHeatmap',
        input_size=input_size,
        heatmap_size=heatmap_size,
        sigma=2,
    )

    cfg_dict = dict(
        default_scope='mmpose',
        model=dict(
            type='TopdownPoseEstimator',
            data_preprocessor=dict(
                type='PoseDataPreprocessor',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                bgr_to_rgb=True,
            ),
            backbone=dict(
                type='HRNet',
                in_channels=3,
                extra=dict(
                    stage1=dict(
                        num_modules=1, num_branches=1, block='BOTTLENECK',
                        num_blocks=(4,), num_channels=(64,)),
                    stage2=dict(
                        num_modules=1, num_branches=2, block='BASIC',
                        num_blocks=(4, 4), num_channels=(32, 64)),
                    stage3=dict(
                        num_modules=4, num_branches=3, block='BASIC',
                        num_blocks=(4, 4, 4), num_channels=(32, 64, 128)),
                    stage4=dict(
                        num_modules=3, num_branches=4, block='BASIC',
                        num_blocks=(4, 4, 4, 4), num_channels=(32, 64, 128, 256)),
                ),
            ),
            head=dict(
                type='HeatmapHead',
                in_channels=32,
                out_channels=133,
                deconv_out_channels=None,
                loss=dict(type='KeypointMSELoss', use_target_weight=True),
                decoder=codec,
            ),
            test_cfg=dict(
                flip_test=True,
                flip_mode='heatmap',
                shift_heatmap=True,
            ),
        ),
        test_dataloader=dict(
            dataset=dict(
                pipeline=[
                    dict(type='LoadImage'),
                    dict(type='GetBBoxCenterScale'),
                    dict(type='TopdownAffine', input_size=input_size),
                    dict(type='PackPoseInputs'),
                ],
            ),
        ),
    )
    return Config(cfg_dict)


def load_hrnet_model(tag, device='cpu'):
    from mmpose.apis import init_model
    info = HRNET_MODELS[tag]
    cfg = build_hrnet_config(info['input_size'], info['heatmap_size'])
    model = init_model(cfg, info['checkpoint'], device=device)
    model.dataset_meta = COCO_WHOLEBODY_DATASET_META
    print(f"Loaded HRNet-w32 ({tag}) on {device}")
    return model


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

    if HRNET_TAG not in HRNET_MODELS:
        sys.exit(f"[error] Unknown HRNET_TAG: {HRNET_TAG}")

    # Find crop subfolders (skip *_hrnet, *_rtmpose, *_openpose folders)
    frame_folders = sorted(
        p for p in root_dir.iterdir()
        if p.is_dir()
        and not any(p.name.endswith(s) for s in ("_hrnet", "_rtmpose", "_openpose"))
        and any(p.glob("*.jpg"))
    )
    if not frame_folders:
        sys.exit(f"[error] No crop subfolders in '{root_dir}'")

    try:
        from mmpose.apis import inference_topdown
    except ImportError:
        sys.exit("[error] mmpose not installed. Run: pip install mmpose mmengine mmcv")

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[Setup]  Root dir     : {root_dir}")
    print(f"[Setup]  HRNet tag    : {HRNET_TAG}")
    print(f"[Setup]  Device       : {device}")
    print(f"[Setup]  Crop folders : {len(frame_folders)}")
    print(f"[Setup]  Loading HRNet ...\n")

    model = load_hrnet_model(HRNET_TAG, device=device)

    # CSV header
    csv_header = ["player_id", "person_index"]
    for kp in JOINT_NAMES:
        csv_header += [f"{kp}_x", f"{kp}_y", f"{kp}_c"]

    total_imgs = 0
    total_rows = 0

    for i, folder in enumerate(frame_folders, start=1):
        out_dir = root_dir / f"{folder.name}_hrnet"
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
            stem = crop_path.stem
            H, W = img.shape[:2]

            # Each crop is a single player — use the whole crop as the bbox
            bbox = np.array([[0, 0, W, H]], dtype=np.float32)

            try:
                results = inference_topdown(model, img, bbox, bbox_format='xyxy')
            except Exception as e:
                print(f"    [error] HRNet failed on {crop_path.name}: {e}")
                rows.append([stem, -1] + [""] * (len(JOINT_NAMES) * 3))
                cv2.imwrite(str(out_dir / crop_path.name), img)
                continue

            if not results:
                rows.append([stem, -1] + [""] * (len(JOINT_NAMES) * 3))
                cv2.imwrite(str(out_dir / crop_path.name), img)
                continue

            inst = results[0].pred_instances
            kpts = inst.keypoints
            scs  = inst.keypoint_scores if hasattr(inst, "keypoint_scores") else None

            if isinstance(kpts, torch.Tensor):
                kpts = kpts.cpu().numpy()
            if scs is not None and isinstance(scs, torch.Tensor):
                scs = scs.cpu().numpy()

            kpts = np.asarray(kpts, dtype=np.float32)
            if kpts.ndim == 3:
                kpts = kpts[0]                  # (133, 2)
            if scs is None:
                scs = np.ones(kpts.shape[0], dtype=np.float32)
            else:
                scs = np.asarray(scs, dtype=np.float32)
                if scs.ndim == 2:
                    scs = scs[0]                # (133,)

            kpts15 = kpts[WHOLEBODY133_TO_OUR15]
            scs15  = scs[WHOLEBODY133_TO_OUR15]

            row = [stem, 0]
            for (x, y), c in zip(kpts15, scs15):
                row += [round(float(x), 3), round(float(y), 3), round(float(c), 4)]
            rows.append(row)

            rendered = draw_15_skeleton(img.copy(), kpts15, scs15, KPT_THRESH)
            cv2.imwrite(str(out_dir / crop_path.name), rendered)

        # One CSV per frame
        csv_path = out_dir / f"{folder.name}_keypoints.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(csv_header)
            w.writerows(rows)
        total_rows += len(rows)
        print(f"        -> {len(rows)} row(s) in {csv_path.name}")

    print("\n-----------------------------------------")
    print("  HRNET SUMMARY")
    print("-----------------------------------------")
    print(f"  Folders processed : {len(frame_folders)}")
    print(f"  Total crops fed   : {total_imgs}")
    print(f"  Total CSV rows    : {total_rows}")
    print("-----------------------------------------\n")


if __name__ == "__main__":
    main()
