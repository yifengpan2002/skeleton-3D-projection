"""
visualize_tvcalib_segmentation.py
-----------------------------------
Shows what TVCalib's segmentation model detected on your NRL frames.

This reveals whether the model found actual rugby field lines or hallucinated
soccer-specific patterns (center circles, penalty boxes that don't exist in rugby).
"""

import sys
import os
import numpy as np
import cv2
import torch
from pathlib import Path
from tqdm import tqdm

# ── TVCalib imports ──────────────────────────────────────────────────────────
sys.path.insert(0, "tvcalib")
sys.path.insert(0, "tvcalib/sn_segmentation/src")

from tvcalib.inference import InferenceDatasetSegmentation, InferenceSegmentationModel
from tvcalib.sncalib_dataset import custom_list_collate


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_NAME      = "ARG_CRO_220001 (1)"
FRAMES_ROOT     = "data/frames"
OUTPUT_ROOT     = "data/detections"
SEG_CHECKPOINT  = "tvcalib/data/segment_localization/train_59.pt"
IMAGE_WIDTH     = 1280
IMAGE_HEIGHT    = 720
NUM_FRAMES      = 10        # visualize first N frames
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Soccer field line class colors (from SoccerNet)
# Class 0 = background, 1-13 = different line types
LINE_COLORS = {
    0:  (0, 0, 0),         # Background - black
    1:  (255, 0, 0),       # Big rect. left bottom - blue
    2:  (0, 255, 0),       # Big rect. left top - green
    3:  (0, 0, 255),       # Big rect. right bottom - red
    4:  (255, 255, 0),     # Big rect. right top - cyan
    5:  (255, 0, 255),     # Circle central - magenta
    6:  (0, 255, 255),     # Circle left - yellow
    7:  (128, 0, 0),       # Circle right - dark blue
    8:  (0, 128, 0),       # Goal left crossbar - dark green
    9:  (0, 0, 128),       # Goal left post left - dark red
    10: (128, 128, 0),     # Goal left post right - dark cyan
    11: (128, 0, 128),     # Goal right crossbar - dark magenta
    12: (0, 128, 128),     # Goal right post left - dark yellow
    13: (192, 192, 192),   # Middle line - gray
}
# ─────────────────────────────────────────────────────────────────────────────


def colorize_segmentation(seg_mask: np.ndarray, colors: dict) -> np.ndarray:
    """
    Convert grayscale segmentation mask to RGB using color map.
    
    Args:
        seg_mask: (H, W) uint8 array with class IDs
        colors: dict mapping class_id -> (B, G, R)
    
    Returns:
        (H, W, 3) RGB colored mask
    """
    h, w = seg_mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    
    for class_id, color in colors.items():
        mask = (seg_mask == class_id)
        colored[mask] = color
    
    return colored


def overlay_segmentation(frame: np.ndarray, seg_colored: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """
    Overlay colored segmentation mask on original frame.
    
    Args:
        frame: (H, W, 3) original BGR image
        seg_colored: (H_seg, W_seg, 3) colored segmentation mask
        alpha: transparency (0 = only frame, 1 = only segmentation)
    
    Returns:
        (H, W, 3) blended image
    """
    # Resize segmentation to match frame
    seg_resized = cv2.resize(seg_colored, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Blend
    blended = cv2.addWeighted(frame, 1 - alpha, seg_resized, alpha, 0)
    
    return blended


def main():
    frames_dir = os.path.join(FRAMES_ROOT, VIDEO_NAME)
    output_dir = os.path.join(OUTPUT_ROOT, VIDEO_NAME, "segmentation_viz")
    
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[Setup]  Frames dir    : {frames_dir}")
    print(f"[Setup]  Output dir    : {output_dir}")
    print(f"[Setup]  Device        : {DEVICE}")
    print(f"[Setup]  Visualizing   : first {NUM_FRAMES} frames\n")
    
    # ═════════════════════════════════════════════════════════════════════════
    #  Load segmentation model
    # ═════════════════════════════════════════════════════════════════════════
    print("[Seg]    Loading segmentation model...")
    
    dataset_seg = InferenceDatasetSegmentation(
        Path(frames_dir), IMAGE_WIDTH, IMAGE_HEIGHT
    )
    
    # Only process first NUM_FRAMES
    dataset_seg = torch.utils.data.Subset(dataset_seg, range(NUM_FRAMES))

    dataloader_seg = torch.utils.data.DataLoader(
        dataset_seg,
        batch_size=4,
        num_workers=0,
        shuffle=False,
        collate_fn=custom_list_collate,
    )
    
    model_seg = InferenceSegmentationModel(SEG_CHECKPOINT, DEVICE)
    print("[Seg]    Model loaded.\n")
    
    # ═════════════════════════════════════════════════════════════════════════
    #  Run segmentation and visualize
    # ═════════════════════════════════════════════════════════════════════════
    print("[Viz]    Running segmentation and creating visualizations...\n")
    
    frame_idx = 0
    for batch_dict in tqdm(dataloader_seg, desc="Processing batches"):
        # Run segmentation
        with torch.no_grad():
            sem_lines = model_seg.inference(batch_dict["image"].to(DEVICE))
        sem_lines = sem_lines.cpu().numpy().astype(np.uint8)  # [B, 256, 455]
        
        # Process each frame in batch
        for i, (image_id, seg_mask) in enumerate(zip(batch_dict["image_id"], sem_lines)):
            # Load original frame
            frame_path = os.path.join(frames_dir, image_id)
            frame = cv2.imread(frame_path)
            
            # Colorize segmentation
            seg_colored = colorize_segmentation(seg_mask, LINE_COLORS)
            
            # Create visualizations
            # 1. Segmentation only (resized to frame size for clarity)
            seg_large = cv2.resize(seg_colored, (frame.shape[1], frame.shape[0]), 
                                   interpolation=cv2.INTER_NEAREST)
            
            # 2. Overlay on frame
            overlay = overlay_segmentation(frame, seg_colored, alpha=0.6)
            
            # 3. Side-by-side comparison
            h, w = frame.shape[:2]
            comparison = np.zeros((h, w * 2, 3), dtype=np.uint8)
            comparison[:, :w] = frame
            comparison[:, w:] = overlay
            
            # Add labels
            cv2.putText(comparison, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(comparison, "Segmentation Overlay", (w + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Count detected line pixels
            line_pixels = np.sum(seg_mask > 0)
            total_pixels = seg_mask.shape[0] * seg_mask.shape[1]
            line_percentage = (line_pixels / total_pixels) * 100
            
            info_text = f"Detected {line_pixels} line pixels ({line_percentage:.2f}%)"
            cv2.putText(comparison, info_text, (10, h - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save visualizations
            frame_stem = Path(image_id).stem
            cv2.imwrite(os.path.join(output_dir, f"{frame_stem}_comparison.jpg"), comparison)
            cv2.imwrite(os.path.join(output_dir, f"{frame_stem}_seg_only.jpg"), seg_large)
            cv2.imwrite(os.path.join(output_dir, f"{frame_stem}_overlay.jpg"), overlay)
            
            frame_idx += 1
    
    print(f"\n[Done]   Saved {frame_idx} visualizations to '{output_dir}'")
    print(f"\n[Interpret] Look at the visualizations:")
    print(f"  • If colored regions align with NRL field lines → segmentation worked")
    print(f"  • If colored blobs are random/in crowd → segmentation hallucinated")
    print(f"  • If mostly black (no color) → model found nothing")
    print(f"\nColors represent different soccer line classes:")
    print(f"  Blue/Green/Red/Cyan   = penalty box lines")
    print(f"  Magenta/Yellow        = circles")
    print(f"  Gray                  = center line")
    print(f"  (NRL fields don't have circles or penalty boxes!)\n")


if __name__ == "__main__":
    main()
