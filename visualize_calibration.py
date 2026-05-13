"""
visualize_calibration.py
-------------------------
Visualizes TVCalib results by projecting field lines back onto frames.

Shows whether the homographies correctly map to the NRL field geometry.
"""

import cv2
import os
import numpy as np
from pathlib import Path


# ─────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────
VIDEO_NAME  = "Adam_Doueihi-Tigers_v_Eels_NRL_R6_2023"
FRAMES_ROOT = "data/frames"
OUTPUT_ROOT = "data/detections"
VIS_OUTPUT  = "data/detections"  # where to save visualizations

# How many frames to visualize (set to None for all)
NUM_FRAMES_TO_VIZ = 10  # visualize first 10 frames

# NRL field lines to draw (in metres, origin = left try-line / top sideline)
FIELD_LINES = {
    # Try lines
    "left_try_line":    [(0, 0), (0, 68)],
    "right_try_line":   [(100, 0), (100, 68)],
    
    # Sidelines
    "top_sideline":     [(0, 0), (100, 0)],
    "bottom_sideline":  [(0, 68), (100, 68)],
    
    # 10m lines
    "10m_left":         [(10, 0), (10, 68)],
    "10m_right":        [(90, 0), (90, 68)],
    
    # 20m lines  
    "20m_left":         [(20, 0), (20, 68)],
    "20m_right":        [(80, 0), (80, 68)],
    
    # Halfway line
    "halfway":          [(50, 0), (50, 68)],
    
    # 40m lines
    "40m_left":         [(40, 0), (40, 68)],
    "40m_right":        [(60, 0), (60, 68)],
}
# ─────────────────────────────────────────


def load_homographies_npz(npz_path: str) -> dict[str, np.ndarray]:
    """Load homographies from NPZ."""
    data = np.load(npz_path, allow_pickle=False)
    frame_names  = data["frame_names"]
    homographies = data["homographies"]
    
    result = {}
    for name, H in zip(frame_names, homographies):
        result[str(name)] = None if np.any(np.isnan(H)) else H.astype(np.float32)
    
    return result


def project_world_to_pixel(world_points: np.ndarray, H: np.ndarray) -> np.ndarray:
    """
    Project world coordinates to pixel coordinates using inverse homography.
    
    Args:
        world_points: (N, 2) array of world coords in metres
        H: (3, 3) homography matrix (world → pixel)
    
    Returns:
        (N, 2) array of pixel coordinates
    """
    # Invert homography to get pixel → world, then invert again for world → pixel
    H_inv = np.linalg.inv(H)
    
    # Convert to homogeneous coordinates
    world_homo = np.concatenate([world_points, np.ones((len(world_points), 1))], axis=1)
    
    # Project
    pixel_homo = (H_inv @ world_homo.T).T
    pixel_coords = pixel_homo[:, :2] / pixel_homo[:, 2:3]
    
    return pixel_coords.astype(np.int32)


def draw_field_lines(frame: np.ndarray, H: np.ndarray, field_lines: dict) -> np.ndarray:
    """
    Draw field lines on a frame using homography.
    
    Args:
        frame: BGR image
        H: homography matrix
        field_lines: dict of {name: [(x1, y1), (x2, y2)]} in world coords
    
    Returns:
        frame with field lines drawn
    """
    vis = frame.copy()
    h, w = frame.shape[:2]
    
    for name, (p1, p2) in field_lines.items():
        # Project world line endpoints to pixel space
        world_pts = np.array([p1, p2], dtype=np.float32)
        pixel_pts = project_world_to_pixel(world_pts, H)
        
        pt1 = tuple(pixel_pts[0])
        pt2 = tuple(pixel_pts[1])
        
        # Only draw if both points are within frame bounds (with margin)
        margin = 100
        if (all(-margin < p[0] < w + margin and -margin < p[1] < h + margin 
                for p in [pt1, pt2])):
            
            # Color coding: try lines red, sidelines blue, others green
            if "try_line" in name:
                color = (0, 0, 255)      # Red
                thickness = 3
            elif "sideline" in name:
                color = (255, 0, 0)      # Blue
                thickness = 3
            else:
                color = (0, 255, 0)      # Green
                thickness = 2
            
            cv2.line(vis, pt1, pt2, color, thickness)
            
            # Add label
            label_pos = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
            cv2.putText(vis, name, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                       0.4, color, 1, cv2.LINE_AA)
    
    return vis


def main():
    frames_dir = os.path.join(FRAMES_ROOT, VIDEO_NAME)
    output_dir = os.path.join(OUTPUT_ROOT, VIDEO_NAME)
    npz_path   = os.path.join(output_dir, "homographies.npz")
    vis_dir    = os.path.join(VIS_OUTPUT, VIDEO_NAME, "calibration_viz")
    
    Path(vis_dir).mkdir(parents=True, exist_ok=True)
    
    print(f"[Viz]    Video name : {VIDEO_NAME}")
    print(f"[Viz]    NPZ file   : {npz_path}")
    print(f"[Viz]    Output     : {vis_dir}\n")
    
    # Load homographies
    homography_map = load_homographies_npz(npz_path)
    print(f"[Viz]    Loaded {len(homography_map)} homographies\n")
    
    # Get frame paths
    frame_paths = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if NUM_FRAMES_TO_VIZ:
        frame_paths = frame_paths[:NUM_FRAMES_TO_VIZ]
    
    print(f"[Viz]    Visualizing {len(frame_paths)} frames...")
    
    for i, frame_path in enumerate(frame_paths):
        frame_stem = frame_path.stem
        H = homography_map.get(frame_stem)
        
        if H is None:
            print(f"  [{i+1}/{len(frame_paths)}] {frame_path.name} - no homography, skipping")
            continue
        
        # Load frame
        frame = cv2.imread(str(frame_path))
        
        # Draw field lines
        vis_frame = draw_field_lines(frame, H, FIELD_LINES)
        
        # Add info text
        info_text = f"Frame: {frame_stem} | Homography valid"
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis_frame, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
        
        # Save
        out_path = os.path.join(vis_dir, f"{frame_stem}_calibration.jpg")
        cv2.imwrite(out_path, vis_frame)
        
        print(f"  [{i+1}/{len(frame_paths)}] {frame_path.name} -> {Path(out_path).name}")
    
    print(f"\n[Done]   Visualizations saved to '{vis_dir}'")
    print(f"         Check if field lines align with the actual field in the frames.")
    print(f"         Red = try lines, Blue = sidelines, Green = other lines\n")


if __name__ == "__main__":
    main()
