"""
auto_field_calibration.py
---------------------------
Automatic field calibration using OpenCV line detection.
No ML, no training data needed - works on any rugby video with white lines on green grass.

Pipeline:
    1. HSV color segmentation → isolate green field
    2. Canny edge detection → find edges
    3. Hough line transform → detect straight lines
    4. Filter lines to field only
    5. Match detected lines to NRL field template
    6. Compute homography via RANSAC
    7. Save to homographies.npz

Processes all frames to handle camera movement.
"""

import cv2
import os
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional
import json


# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
VIDEO_PATH  = "data/clips-NRL/Adam_Doueihi-Tigers_v_Eels_NRL_R6_2023.mkv"
FRAMES_ROOT = "data/frames"
OUTPUT_ROOT = "data/detections"

# HSV range for green grass field (tune these if needed for different lighting)
GREEN_LOWER = np.array([35, 40, 40])     # Lower bound (H, S, V)
GREEN_UPPER = np.array([85, 255, 255])   # Upper bound

# Hough line detection parameters
CANNY_LOW = 50
CANNY_HIGH = 150
HOUGH_THRESHOLD = 80        # Minimum votes for a line
HOUGH_MIN_LINE_LENGTH = 100 # Minimum line length in pixels
HOUGH_MAX_LINE_GAP = 20     # Maximum gap between line segments

# NRL field template (in metres, origin = left try-line / top sideline)
# Key lines that are usually visible in broadcast
NRL_TEMPLATE_LINES = {
    "left_try": [(0, 0), (0, 68)],
    "right_try": [(100, 0), (100, 68)],
    "top_sideline": [(0, 0), (100, 0)],
    "bottom_sideline": [(0, 68), (100, 68)],
    "10m_left": [(10, 0), (10, 68)],
    "10m_right": [(90, 0), (90, 68)],
    "halfway": [(50, 0), (50, 68)],
}

# How many frames to sample for calibration (process every N-th frame)
CALIBRATION_FRAME_STEP = 5  # Calibrate every 5th frame
MIN_LINES_REQUIRED = 4      # Minimum lines needed for reliable calibration
# ─────────────────────────────────────────────────────────────────────────────


def detect_field_mask(frame: np.ndarray) -> np.ndarray:
    """
    Detect green field area using HSV color segmentation.
    
    Returns binary mask where 255 = field, 0 = not field
    """
    # Convert to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Threshold for green
    mask = cv2.inRange(hsv, GREEN_LOWER, GREEN_UPPER)
    
    # Morphological operations to clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Fill holes
    
    # Keep only the largest connected component (the field)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if num_labels > 1:
        # Find largest component (excluding background=0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
    
    return mask


def detect_lines_hough(frame: np.ndarray, field_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
    """
    Detect white lines on the field using Canny + Hough transform.
    
    Returns list of lines as [(x1, y1, x2, y2), ...]
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply field mask
    masked_gray = cv2.bitwise_and(gray, gray, mask=field_mask)
    
    # Edge detection
    edges = cv2.Canny(masked_gray, CANNY_LOW, CANNY_HIGH, apertureSize=3)
    
    # Hough line detection (probabilistic)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LINE_LENGTH,
        maxLineGap=HOUGH_MAX_LINE_GAP,
    )
    
    if lines is None:
        return []
    
    # Convert to list of (x1, y1, x2, y2)
    lines_list = [tuple(line[0]) for line in lines]
    
    return lines_list


def filter_lines_by_orientation(lines: List[Tuple[int, int, int, int]]) -> dict:
    """
    Group lines into horizontal and vertical based on angle.
    
    Returns {'horizontal': [...], 'vertical': [...]}
    """
    horizontal = []
    vertical = []
    
    for x1, y1, x2, y2 in lines:
        dx = x2 - x1
        dy = y2 - y1
        angle = np.abs(np.arctan2(dy, dx) * 180 / np.pi)
        
        if angle < 30 or angle > 150:  # Nearly horizontal
            horizontal.append((x1, y1, x2, y2))
        elif 60 < angle < 120:          # Nearly vertical
            vertical.append((x1, y1, x2, y2))
    
    return {'horizontal': horizontal, 'vertical': vertical}


def merge_collinear_lines(lines: List[Tuple[int, int, int, int]], max_distance: float = 10) -> List[Tuple[int, int, int, int]]:
    """
    Merge nearby collinear line segments into single lines.
    """
    if not lines:
        return []
    
    merged = []
    used = [False] * len(lines)
    
    for i, line1 in enumerate(lines):
        if used[i]:
            continue
        
        x1, y1, x2, y2 = line1
        group = [line1]
        
        for j, line2 in enumerate(lines[i + 1:], start=i + 1):
            if used[j]:
                continue
            
            x3, y3, x4, y4 = line2
            
            # Check if lines are close and roughly parallel
            # (simplified check - just distance between midpoints)
            mid1 = ((x1 + x2) / 2, (y1 + y2) / 2)
            mid2 = ((x3 + x4) / 2, (y3 + y4) / 2)
            dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
            
            if dist < max_distance:
                group.append(line2)
                used[j] = True
        
        # Merge group into single line spanning the extents
        all_x = [x for line in group for x in [line[0], line[2]]]
        all_y = [y for line in group for y in [line[1], line[3]]]
        merged_line = (min(all_x), min(all_y), max(all_x), max(all_y))
        merged.append(merged_line)
        used[i] = True
    
    return merged


def match_lines_to_template(detected_lines: dict, template: dict, frame_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """
    Match detected lines to NRL field template and compute homography.
    
    Uses RANSAC to find best correspondence between detected and template lines.
    
    Returns (3, 3) homography matrix or None if matching fails.
    """
    h, w = frame_shape[:2]
    
    # Extract template points
    template_points = []
    for line_pts in template.values():
        template_points.extend(line_pts)
    template_points = np.array(template_points, dtype=np.float32)
    
    # Sample points from detected lines
    detected_points = []
    for orientation in ['horizontal', 'vertical']:
        for x1, y1, x2, y2 in detected_lines[orientation]:
            # Sample points along the line
            num_samples = 5
            for t in np.linspace(0, 1, num_samples):
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                detected_points.append([px, py])
    
    if len(detected_points) < 4:
        return None
    
    detected_points = np.array(detected_points, dtype=np.float32)
    
    # Try to find homography via RANSAC
    # We need at least 4 point correspondences
    # Since we don't know exact correspondences, we'll try a simpler approach:
    # Assume field is roughly centered and try standard 4-corner match
    
    # Fallback: if we have good horizontal and vertical lines,
    # estimate field corners from line intersections
    corners_pixel = estimate_field_corners(detected_lines, w, h)
    if corners_pixel is None:
        return None
    
    # Template corners (try-lines × sidelines)
    corners_world = np.float32([
        [0, 0],      # top-left
        [100, 0],    # top-right
        [100, 68],   # bottom-right
        [0, 68],     # bottom-left
    ])
    
    H, status = cv2.findHomography(corners_pixel, corners_world, cv2.RANSAC, 5.0)
    return H


def estimate_field_corners(detected_lines: dict, width: int, height: int) -> Optional[np.ndarray]:
    """
    Estimate field corner positions from detected horizontal and vertical lines.
    
    Returns (4, 2) array of corner pixel coordinates or None if insufficient lines.
    """
    h_lines = detected_lines['horizontal']
    v_lines = detected_lines['vertical']
    
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None
    
    # Find topmost and bottommost horizontal lines
    h_lines_sorted = sorted(h_lines, key=lambda line: (line[1] + line[3]) / 2)
    top_line = h_lines_sorted[0]
    bottom_line = h_lines_sorted[-1]
    
    # Find leftmost and rightmost vertical lines
    v_lines_sorted = sorted(v_lines, key=lambda line: (line[0] + line[2]) / 2)
    left_line = v_lines_sorted[0]
    right_line = v_lines_sorted[-1]
    
    # Compute intersections
    def line_intersection(line1, line2):
        x1, y1, x2, y2 = line1
        x3, y3, x4, y4 = line2
        
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-10:
            return None
        
        px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denom
        py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denom
        
        return (int(px), int(py))
    
    tl = line_intersection(top_line, left_line)
    tr = line_intersection(top_line, right_line)
    br = line_intersection(bottom_line, right_line)
    bl = line_intersection(bottom_line, left_line)
    
    if any(corner is None for corner in [tl, tr, br, bl]):
        return None
    
    corners = np.float32([tl, tr, br, bl])
    return corners


def calibrate_frame(frame: np.ndarray, debug: bool = False) -> Optional[np.ndarray]:
    """
    Calibrate a single frame and return homography matrix.
    
    Args:
        frame: BGR image
        debug: if True, return debug visualization info
    
    Returns:
        (3, 3) homography matrix or None if calibration fails
    """
    # Step 1: Detect field
    field_mask = detect_field_mask(frame)
    
    # Step 2: Detect lines
    raw_lines = detect_lines_hough(frame, field_mask)
    
    if len(raw_lines) < MIN_LINES_REQUIRED:
        return None
    
    # Step 3: Filter and group by orientation
    oriented_lines = filter_lines_by_orientation(raw_lines)
    
    # Step 4: Merge collinear segments
    oriented_lines['horizontal'] = merge_collinear_lines(oriented_lines['horizontal'])
    oriented_lines['vertical'] = merge_collinear_lines(oriented_lines['vertical'])
    
    # Step 5: Match to template and compute homography
    H = match_lines_to_template(oriented_lines, NRL_TEMPLATE_LINES, frame.shape)
    
    return H


def save_homographies_npz(
    homographies: dict[str, Optional[np.ndarray]],
    frame_names: List[str],
    output_path: str,
) -> None:
    """
    Save homographies to NPZ. Fill failed frames with NaN.
    """
    H_list = []
    for name in frame_names:
        H = homographies.get(name)
        if H is None:
            H_list.append(np.full((3, 3), np.nan, dtype=np.float64))
        else:
            H_list.append(H.astype(np.float64))
    
    H_array = np.stack(H_list, axis=0)
    names_array = np.array(frame_names)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, frame_names=names_array, homographies=H_array)
    
    valid = sum(1 for h in homographies.values() if h is not None)
    print(f"\n[NPZ]    Saved {len(frame_names)} homographies")
    print(f"         Valid: {valid} | Failed: {len(frame_names) - valid}")
    print(f"         Output: {output_path}")


def main():
    video_name = Path(VIDEO_PATH).stem
    frames_dir = os.path.join(FRAMES_ROOT, video_name)
    output_dir = os.path.join(OUTPUT_ROOT, video_name)
    npz_path   = os.path.join(output_dir, "homographies.npz")
    
    print("=" * 60)
    print("  AUTOMATIC FIELD CALIBRATION (OpenCV)")
    print("=" * 60)
    print(f"  Video      : {VIDEO_PATH}")
    print(f"  Frames dir : {frames_dir}")
    print(f"  Output     : {npz_path}")
    print(f"  Method     : HSV green + Canny + Hough + template matching")
    print("=" * 60 + "\n")
    
    # Get all frames
    frame_files = sorted(Path(frames_dir).glob("frame_*.jpg"))
    if not frame_files:
        raise FileNotFoundError(f"No frames in '{frames_dir}'. Run extract_frame.py first.")
    
    print(f"[Frames] Found {len(frame_files)} frames")
    print(f"[Calibrate] Processing every {CALIBRATION_FRAME_STEP}-th frame...\n")
    
    # Calibrate sampled frames
    homographies = {}
    frame_names = []
    
    for i, frame_file in enumerate(frame_files):
        frame_name = frame_file.stem
        frame_names.append(frame_name)
        
        # Only calibrate every N-th frame
        if i % CALIBRATION_FRAME_STEP != 0:
            homographies[frame_name] = None  # Will interpolate later
            continue
        
        frame = cv2.imread(str(frame_file))
        H = calibrate_frame(frame)
        homographies[frame_name] = H
        
        status = "✓" if H is not None else "✗"
        if (i + 1) % 50 == 0 or (i + 1) == len(frame_files):
            print(f"  [{i + 1}/{len(frame_files)}] {frame_name} {status}")
    
    # Interpolate failed frames from nearby successful ones
    print(f"\n[Interpolate] Filling gaps...")
    homographies = interpolate_homographies(homographies, frame_names)
    
    # Save
    save_homographies_npz(homographies, frame_names, npz_path)
    
    print(f"\n[Done] Calibration complete!")
    print(f"       Run 'python visualize_calibration.py' to verify.")


def interpolate_homographies(
    homographies: dict[str, Optional[np.ndarray]],
    frame_names: List[str],
) -> dict[str, Optional[np.ndarray]]:
    """
    Fill missing homographies by copying from nearest successful frame.
    """
    # Find all successful calibrations
    valid_indices = [i for i, name in enumerate(frame_names) if homographies[name] is not None]
    
    if not valid_indices:
        print("  [Warning] No valid calibrations found!")
        return homographies
    
    # Fill gaps
    filled = 0
    for i, name in enumerate(frame_names):
        if homographies[name] is None:
            # Find nearest valid frame
            nearest_idx = min(valid_indices, key=lambda x: abs(x - i))
            nearest_name = frame_names[nearest_idx]
            homographies[name] = homographies[nearest_name].copy()
            filled += 1
    
    print(f"  Filled {filled} frames from nearest neighbors")
    return homographies


if __name__ == "__main__":
    main()
