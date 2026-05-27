"""
field_homography.py
--------------------
Step 4: Homography Matrix Estimation for Rugby Fields

Pipeline:
1. Detect field lines (reuses detection logic)
2. Find intersections of detected lines
3. Interactive GUI: click intersections → assign to rugby template points
4. Compute homography matrix (H)
5. Visualize by projecting rugby field template back onto image

Usage:
    python field_homography.py -i "data/simple_rugby_field/image.jpg"
    python field_homography.py -i "data/simple_rugby_field"

References:
    - World Rugby Laws: https://passport.world.rugby/laws-of-the-game/
    - Rugby Union field: 100m (try-to-try) x 70m (touch-to-touch)
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


# ========================================================================
# RUGBY FIELD TEMPLATE (World Rugby dimensions in metres)
# ========================================================================
# Origin = left try-line / top touchline corner
# X axis = along the field (0 to 100m, try-line to try-line)
# Y axis = across the field (0 to 70m, touchline to touchline)

RUGBY_FIELD_POINTS = {
    # Format: "name": (x_metres, y_metres)
    # Try lines (goal lines)
    "try_left_top":       (0,   0),
    "try_left_bottom":    (0,   70),
    # 22m lines
    "22m_left_top":       (22,  0),
    "22m_left_bottom":    (22,  70),
    # 10m lines
    "10m_left_top":       (40,  0),
    "10m_left_bottom":    (40,  70),
    # Halfway line
    "halfway_top":        (50,  0),
    "halfway_bottom":     (50,  70),
    # 10m right
    "10m_right_top":      (60,  0),
    "10m_right_bottom":   (60,  70),
    # 22m right
    "22m_right_top":      (78,  0),
    "22m_right_bottom":   (78,  70),
    # Try line right
    "try_right_top":      (100, 0),
    "try_right_bottom":   (100, 70),
}

# Field lines for visualization (pairs of template point names)
RUGBY_FIELD_LINES = [
    # Touchlines (sidelines)
    ("try_left_top",    "try_right_top"),
    ("try_left_bottom", "try_right_bottom"),
    # Try lines
    ("try_left_top",    "try_left_bottom"),
    ("try_right_top",   "try_right_bottom"),
    # 22m lines
    ("22m_left_top",    "22m_left_bottom"),
    ("22m_right_top",   "22m_right_bottom"),
    # 10m lines
    ("10m_left_top",    "10m_left_bottom"),
    ("10m_right_top",   "10m_right_bottom"),
    # Halfway
    ("halfway_top",     "halfway_bottom"),
]


# ========================================================================
# LINE DETECTION (same as your working pipeline)
# ========================================================================

def detect_lines(frame):
    """Detect field lines using green mask + edge subtraction + Hough."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Green mask
    lower_green = np.array([35, 25, 25])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    kernel = np.ones((5, 5), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    green_field = cv2.bitwise_and(frame, frame, mask=green_mask)

    # White line detection
    gray_green = cv2.cvtColor(green_field, cv2.COLOR_BGR2GRAY)
    white_lines = cv2.adaptiveThreshold(
        gray_green, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, -5
    )

    line_kernel = np.ones((3, 3), np.uint8)
    white_lines_cleaned = cv2.morphologyEx(white_lines, cv2.MORPH_OPEN, line_kernel)
    connect_kernel = np.ones((5, 5), np.uint8)
    white_lines_cleaned = cv2.morphologyEx(white_lines_cleaned, cv2.MORPH_CLOSE, connect_kernel)

    # Edge subtraction: green_field edges - green_mask boundary
    edges_all = cv2.Canny(green_field, 50, 150)
    edges_boundary = cv2.Canny(green_mask, 50, 150)
    boundary_kernel = np.ones((5, 5), np.uint8)
    edges_boundary_dilated = cv2.dilate(edges_boundary, boundary_kernel, iterations=1)
    edges = cv2.subtract(edges_all, edges_boundary_dilated)

    # Hough lines
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180,
        threshold=80, minLineLength=60, maxLineGap=25,
    )

    return lines, green_mask


# ========================================================================
# INTERSECTION FINDING
# ========================================================================

def line_intersection(p1, p2, p3, p4):
    """
    Find intersection of line (p1-p2) and line (p3-p4).
    Returns (x, y) or None if parallel.
    """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-6:
        return None  # Parallel

    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    px = x1 + t * (x2 - x1)
    py = y1 + t * (y2 - y1)

    return (px, py)


def merge_collinear_lines(lines, angle_thresh=8, dist_thresh=15):
    """
    Merge Hough segments that lie on the same physical line.
    
    Two segments are merged if:
    - Their angles are within `angle_thresh` degrees of each other
    - The perpendicular distance between them is below `dist_thresh` pixels
    
    Result: each physical field line becomes ONE merged line, eliminating
    "phantom intersections" between overlapping Hough detections.
    
    Returns: list of merged lines in same format as Hough output [[(x1,y1,x2,y2)], ...]
    """
    if lines is None or len(lines) == 0:
        return []
    
    # Parse all segments
    segments = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        segments.append({
            'pts': (x1, y1, x2, y2),
            'angle': angle,
            'length': length,
            'merged': False,
        })
    
    merged_lines = []
    
    for i, seg_i in enumerate(segments):
        if seg_i['merged']:
            continue
        
        group_pts = [(seg_i['pts'][0], seg_i['pts'][1]),
                     (seg_i['pts'][2], seg_i['pts'][3])]
        seg_i['merged'] = True
        
        for j, seg_j in enumerate(segments):
            if seg_j['merged']:
                continue
            
            # Check angle similarity
            angle_diff = abs(seg_i['angle'] - seg_j['angle'])
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff > angle_thresh:
                continue
            
            # Check perpendicular distance from seg_j to line of seg_i
            x1, y1, x2, y2 = seg_i['pts']
            dx, dy = x2 - x1, y2 - y1
            line_len = max(np.sqrt(dx**2 + dy**2), 1)
            
            # Distance from midpoint of seg_j to seg_i's line
            mid_x = (seg_j['pts'][0] + seg_j['pts'][2]) / 2
            mid_y = (seg_j['pts'][1] + seg_j['pts'][3]) / 2
            perp_dist = abs((mid_x - x1) * dy - (mid_y - y1) * dx) / line_len
            
            if perp_dist < dist_thresh:
                group_pts.append((seg_j['pts'][0], seg_j['pts'][1]))
                group_pts.append((seg_j['pts'][2], seg_j['pts'][3]))
                seg_j['merged'] = True
        
        # Fit one line through all endpoints in the group using PCA
        if len(group_pts) >= 2:
            pts_arr = np.array(group_pts, dtype=np.float64)
            mean = pts_arr.mean(axis=0)
            centered = pts_arr - mean
            _, _, Vt = np.linalg.svd(centered)
            direction = Vt[0]
            
            # Project all points onto the line direction and take extremes
            projections = centered @ direction
            p1 = mean + projections.min() * direction
            p2 = mean + projections.max() * direction
            
            merged_lines.append([[int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1])]])
    
    return np.array(merged_lines) if merged_lines else None


def point_to_segment_distance(px, py, x1, y1, x2, y2):
    """Distance from point (px, py) to the segment defined by (x1,y1)-(x2,y2)."""
    dx, dy = x2 - x1, y2 - y1
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq < 1e-6:
        return np.sqrt((px - x1)**2 + (py - y1)**2)
    
    # Project point onto segment, clamp to [0, 1]
    t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / seg_len_sq))
    closest_x = x1 + t * dx
    closest_y = y1 + t * dy
    return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)


def find_all_intersections(lines, img_shape, green_mask=None,
                           angle_min=30, max_dist_to_segment=80,
                           merge_first=True):
    """
    Find REAL intersections between detected line pairs.
    
    Three fixes applied:
    1. Merge collinear lines first (eliminates phantom intersections)
    2. Stricter angle threshold (only true perpendicular crossings)
    3. Verify intersection is close to BOTH line segments (not extrapolated)
    
    Parameters:
        lines: Hough lines output
        img_shape: image shape (h, w, ...)
        green_mask: optional binary mask of the field
        angle_min: minimum angle between two lines (default 60°, was 15°)
        max_dist_to_segment: max pixel distance from intersection to each segment
        merge_first: whether to merge collinear segments first
    """
    if lines is None:
        return []
    
    # FIX 1: Merge collinear segments first
    if merge_first:
        lines = merge_collinear_lines(lines, angle_thresh=8, dist_thresh=15)
        if lines is None or len(lines) == 0:
            return []
        print(f"    After merging: {len(lines)} unique lines")
    
    h, w = img_shape[:2]
    margin = 50
    intersections = []
    
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            x1, y1, x2, y2 = lines[i][0]
            x3, y3, x4, y4 = lines[j][0]
            
            # FIX 3: Stricter angle threshold (default 60° instead of 15°)
            angle1 = np.degrees(np.arctan2(y2 - y1, x2 - x1)) % 180
            angle2 = np.degrees(np.arctan2(y4 - y3, x4 - x3)) % 180
            angle_diff = abs(angle1 - angle2)
            if angle_diff > 90:
                angle_diff = 180 - angle_diff
            if angle_diff < angle_min:
                continue
            
            pt = line_intersection((x1, y1), (x2, y2), (x3, y3), (x4, y4))
            if pt is None:
                continue
            
            px, py = pt
            
            # Check bounds
            if px < -margin or px > w + margin or py < -margin or py > h + margin:
                continue
            
            # FIX 2: Intersection must be close to BOTH segments
            # (real crossings happen at/near the segments, not extrapolated far away)
            dist_to_seg1 = point_to_segment_distance(px, py, x1, y1, x2, y2)
            dist_to_seg2 = point_to_segment_distance(px, py, x3, y3, x4, y4)
            
            if dist_to_seg1 > max_dist_to_segment or dist_to_seg2 > max_dist_to_segment:
                continue
            
            # Check if on green field
            if green_mask is not None:
                ix, iy = int(np.clip(px, 0, w - 1)), int(np.clip(py, 0, h - 1))
                if green_mask[iy, ix] == 0:
                    continue
            
            intersections.append((int(px), int(py)))
    
    # Remove duplicate points (within 20px of each other)
    filtered = []
    for pt in intersections:
        is_dup = False
        for existing in filtered:
            dist = np.sqrt((pt[0] - existing[0])**2 + (pt[1] - existing[1])**2)
            if dist < 20:
                is_dup = True
                break
        if not is_dup:
            filtered.append(pt)
    
    return filtered


# ========================================================================
# INTERACTIVE POINT MATCHING (GUI)
# ========================================================================

class PointMatcher:
    """
    Interactive GUI for matching detected intersections to rugby template points.

    Usage:
    1. Click on a detected intersection (yellow circle) to select it
    2. Press the number key shown next to the template point name
    3. Repeat for at least 4 points
    4. Press 'q' to finish and compute homography
    5. Press 'r' to reset all matches
    """
    def __init__(self, img, intersections, template_points):
        self.img = img.copy()
        self.intersections = intersections
        self.template_points = template_points
        self.template_names = list(template_points.keys())

        self.selected_intersection = None
        self.matches = []  # List of (image_pt, world_pt, name)

        self.window_name = "Point Matcher - Click intersections, press number to assign"

    def _draw(self):
        vis = self.img.copy()
        h, w = vis.shape[:2]

        # Draw detected intersections (yellow circles with index)
        for i, pt in enumerate(self.intersections):
            color = (0, 255, 255)  # Yellow
            # Check if already matched
            for match in self.matches:
                if match[0] == pt:
                    color = (0, 255, 0)  # Green = matched
                    break

            cv2.circle(vis, pt, 8, color, 2)
            cv2.putText(vis, str(i), (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Highlight selected intersection
        if self.selected_intersection is not None:
            pt = self.intersections[self.selected_intersection]
            cv2.circle(vis, pt, 15, (0, 0, 255), 3)

        # Draw template point list on the side
        y_offset = 30
        cv2.putText(vis, "Template Points:", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 25

        for i, name in enumerate(self.template_names):
            # Check if already matched
            matched = False
            for match in self.matches:
                if match[2] == name:
                    matched = True
                    break

            color = (0, 255, 0) if matched else (200, 200, 200)
            key_label = chr(ord('a') + i) if i < 26 else str(i)
            text = f"[{key_label}] {name}"
            cv2.putText(vis, text, (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            y_offset += 18

        # Instructions at bottom
        cv2.putText(vis, f"Matches: {len(self.matches)}/4+  |  Click point -> press letter  |  'q'=done  'r'=reset",
                    (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return vis

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Find closest intersection
            min_dist = float('inf')
            closest_idx = None
            for i, pt in enumerate(self.intersections):
                dist = np.sqrt((x - pt[0])**2 + (y - pt[1])**2)
                if dist < min_dist:
                    min_dist = dist
                    closest_idx = i

            if min_dist < 30:  # Click within 30px
                self.selected_intersection = closest_idx
                print(f"  Selected intersection #{closest_idx} at {self.intersections[closest_idx]}")

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

        print("\n" + "=" * 60)
        print("INTERACTIVE POINT MATCHING")
        print("=" * 60)
        print("1. Click on a yellow intersection point")
        print("2. Press the letter key for the template point (a, b, c...)")
        print("3. Repeat for at least 4 points")
        print("4. Press 'q' when done, 'r' to reset")
        print("=" * 60)

        while True:
            vis = self._draw()
            cv2.imshow(self.window_name, vis)
            key = cv2.waitKey(30) & 0xFF

            if key == ord('q'):
                break
            elif key == ord('r'):
                self.matches = []
                self.selected_intersection = None
                print("  Reset all matches")
            elif self.selected_intersection is not None:
                # Map key press to template point index
                idx = -1
                if ord('a') <= key <= ord('z'):
                    idx = key - ord('a')

                if 0 <= idx < len(self.template_names):
                    name = self.template_names[idx]
                    img_pt = self.intersections[self.selected_intersection]
                    world_pt = self.template_points[name]

                    # Remove existing match for this template point
                    self.matches = [m for m in self.matches if m[2] != name]

                    self.matches.append((img_pt, world_pt, name))
                    print(f"  Matched: intersection #{self.selected_intersection} "
                          f"-> {name} ({world_pt[0]}m, {world_pt[1]}m)")
                    self.selected_intersection = None

        cv2.destroyAllWindows()
        return self.matches


# ========================================================================
# AUTOMATIC POINT MATCHING (no GUI needed)
# ========================================================================

class AutoPointMatcher:
    """
    CLI-based point matching for when GUI is not available.
    Shows numbered intersections, user types pairs.
    """
    def __init__(self, img, intersections, template_points, output_dir):
        self.img = img
        self.intersections = intersections
        self.template_points = template_points
        self.template_names = list(template_points.keys())
        self.output_dir = output_dir

    def run(self):
        # Save image with numbered intersections
        vis = self.img.copy()
        for i, pt in enumerate(self.intersections):
            cv2.circle(vis, pt, 8, (0, 255, 255), 2)
            cv2.putText(vis, str(i), (pt[0] + 10, pt[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imwrite(f"{self.output_dir}/intersections_numbered.jpg", vis)

        print("\n" + "=" * 60)
        print("POINT MATCHING (CLI MODE)")
        print("=" * 60)
        print(f"See: {self.output_dir}/intersections_numbered.jpg")
        print(f"\nDetected {len(self.intersections)} intersections (numbered 0-{len(self.intersections)-1})")
        print(f"\nAvailable template points:")
        for i, name in enumerate(self.template_names):
            coords = self.template_points[name]
            print(f"  [{i:2d}] {name:30s} ({coords[0]:5.1f}m, {coords[1]:5.1f}m)")

        print(f"\nEnter matches as: intersection_number template_number")
        print(f"Example: 3 5  (intersection #3 = template point #5)")
        print(f"Type 'done' when finished (need at least 4 matches)")
        print("=" * 60)

        matches = []
        while True:
            try:
                user_input = input(f"  Match {len(matches)+1} > ").strip()
                if user_input.lower() == 'done':
                    break
                parts = user_input.split()
                if len(parts) != 2:
                    print("    Enter two numbers: intersection_idx template_idx")
                    continue

                int_idx = int(parts[0])
                tpl_idx = int(parts[1])

                if int_idx < 0 or int_idx >= len(self.intersections):
                    print(f"    Invalid intersection index (0-{len(self.intersections)-1})")
                    continue
                if tpl_idx < 0 or tpl_idx >= len(self.template_names):
                    print(f"    Invalid template index (0-{len(self.template_names)-1})")
                    continue

                name = self.template_names[tpl_idx]
                img_pt = self.intersections[int_idx]
                world_pt = self.template_points[name]
                matches.append((img_pt, world_pt, name))
                print(f"    ✓ Intersection #{int_idx} -> {name} ({world_pt[0]}m, {world_pt[1]}m)")

            except (ValueError, EOFError):
                break

        return matches


# ========================================================================
# HOMOGRAPHY COMPUTATION
# ========================================================================

def compute_homography(matches):
    """
    Compute homography from matched point pairs.
    Needs at least 4 matches.

    Returns:
        H: (3, 3) homography matrix (image pixels -> world metres)
        None if not enough matches
    """
    if len(matches) < 4:
        print(f"[Error] Need at least 4 matches, got {len(matches)}")
        return None

    img_pts = np.array([m[0] for m in matches], dtype=np.float32)
    world_pts = np.array([m[1] for m in matches], dtype=np.float32)

    H, mask = cv2.findHomography(img_pts, world_pts, cv2.RANSAC, 5.0)

    if H is None:
        print("[Error] Homography computation failed")
        return None

    # Check how many inliers
    inliers = mask.ravel().sum()
    print(f"\n[Homography] Computed from {len(matches)} matches ({inliers} inliers)")

    # Compute reprojection error
    img_pts_homo = np.column_stack([img_pts, np.ones(len(img_pts))])
    projected = (H @ img_pts_homo.T).T
    projected = projected[:, :2] / projected[:, 2:3]
    errors = np.linalg.norm(projected - world_pts, axis=1)
    print(f"[Homography] Mean reprojection error: {errors.mean():.2f}m")
    print(f"[Homography] Max reprojection error:  {errors.max():.2f}m")

    return H


# ========================================================================
# VISUALIZATION: Project field template back onto image
# ========================================================================

def visualize_homography(img, H, output_path):
    """
    Project rugby field lines back onto the image using inverse homography.
    If lines align with actual field lines → calibration is correct.
    """
    vis = img.copy()
    H_inv = np.linalg.inv(H)  # world → image

    def world_to_pixel(wx, wy):
        pt = np.array([wx, wy, 1.0])
        px = H_inv @ pt
        px = px[:2] / px[2]
        return int(px[0]), int(px[1])

    h, w = img.shape[:2]

    # Draw field lines
    for name1, name2 in RUGBY_FIELD_LINES:
        pt1_world = RUGBY_FIELD_POINTS[name1]
        pt2_world = RUGBY_FIELD_POINTS[name2]

        pt1_px = world_to_pixel(*pt1_world)
        pt2_px = world_to_pixel(*pt2_world)

        # Check if within image bounds (with margin)
        margin = 200
        pts_ok = all(
            -margin < p[0] < w + margin and -margin < p[1] < h + margin
            for p in [pt1_px, pt2_px]
        )

        if pts_ok:
            # Color by line type
            if "try" in name1:
                color = (0, 0, 255)    # Red = try lines
                thickness = 3
            elif "22m" in name1 or "22m" in name2:
                color = (255, 0, 0)    # Blue = 22m lines
                thickness = 2
            elif "halfway" in name1:
                color = (0, 255, 0)    # Green = halfway
                thickness = 3
            elif "10m" in name1 or "10m" in name2:
                color = (255, 255, 0)  # Cyan = 10m lines
                thickness = 2
            else:
                color = (255, 255, 255)
                thickness = 2

            cv2.line(vis, pt1_px, pt2_px, color, thickness)

    # Add legend
    legend_y = 30
    for label, color in [("Try lines", (0,0,255)), ("22m lines", (255,0,0)),
                          ("Halfway", (0,255,0)), ("10m lines", (255,255,0))]:
        cv2.putText(vis, label, (w - 200, legend_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        legend_y += 25

    cv2.imwrite(str(output_path), vis)
    print(f"[Viz] Saved: {output_path}")

    return vis


# ========================================================================
# MAIN PIPELINE
# ========================================================================

def process_image(image_path, output_dir, use_gui=True):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Load image
    frame = cv2.imread(str(image_path))
    if frame is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    h, w = frame.shape[:2]
    print(f"\n[Load] {w}x{h}: {Path(image_path).name}")

    # Step 1: Detect lines
    print("[Step 1] Detecting field lines...")
    lines, green_mask = detect_lines(frame)
    line_count = len(lines) if lines is not None else 0
    print(f"    Hough lines: {line_count}")

    if line_count == 0:
        print("[Error] No lines detected. Cannot compute homography.")
        return None

    # Draw detected lines
    line_vis = frame.copy()
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(line_vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite(f"{output_dir}/detected_lines.jpg", line_vis)

    # Step 2: Find intersections
    print("[Step 2] Finding line intersections...")
    intersections = find_all_intersections(lines, frame.shape, green_mask)
    print(f"    Intersections found: {len(intersections)}")

    if len(intersections) < 4:
        print("[Error] Need at least 4 intersections. Try adjusting Hough parameters.")
        return None

    # Draw intersections
    int_vis = frame.copy()
    for i, pt in enumerate(intersections):
        cv2.circle(int_vis, pt, 8, (0, 255, 255), 2)
        cv2.putText(int_vis, str(i), (pt[0] + 10, pt[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.imwrite(f"{output_dir}/intersections.jpg", int_vis)

    # Step 3: Match points
    print("[Step 3] Matching intersections to template...")
    if use_gui:
        matcher = PointMatcher(frame, intersections, RUGBY_FIELD_POINTS)
    else:
        matcher = AutoPointMatcher(frame, intersections, RUGBY_FIELD_POINTS, output_dir)

    matches = matcher.run()

    if len(matches) < 4:
        print(f"[Error] Only {len(matches)} matches. Need at least 4.")
        return None

    # Step 4: Compute homography
    print("[Step 4] Computing homography...")
    H = compute_homography(matches)

    if H is None:
        return None

    # Save homography
    np.save(f"{output_dir}/homography.npy", H)
    print(f"[Save] Homography matrix -> {output_dir}/homography.npy")

    # Step 5: Visualize
    print("[Step 5] Visualizing projected field lines...")
    visualize_homography(frame, H, f"{output_dir}/projected_field.jpg")

    # Save match info
    with open(f"{output_dir}/matches.txt", "w") as f:
        f.write("# image_x image_y world_x world_y name\n")
        for img_pt, world_pt, name in matches:
            f.write(f"{img_pt[0]} {img_pt[1]} {world_pt[0]} {world_pt[1]} {name}\n")

    print(f"\n[Done] All outputs in: {output_dir}/")
    print(f"  detected_lines.jpg   - Hough lines")
    print(f"  intersections.jpg    - Numbered intersection points")
    print(f"  projected_field.jpg  - Rugby template projected onto image")
    print(f"  homography.npy       - H matrix (load with np.load)")
    print(f"  matches.txt          - Point correspondences")

    return H


# ========================================================================
# ENTRY POINT
# ========================================================================

def main():
    parser = argparse.ArgumentParser(description="Rugby field homography estimation")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Image or folder of images")
    parser.add_argument("--output", "-o", type=str, default="outputs/homography",
                        help="Output directory")
    parser.add_argument("--cli", action="store_true",
                        help="Use CLI mode instead of GUI (for headless/SSH)")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_root = Path(args.output)
    IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

    if input_path.is_file():
        image_files = [input_path]
    elif input_path.is_dir():
        image_files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    else:
        raise FileNotFoundError(f"Not found: {input_path}")

    print("=" * 60)
    print("RUGBY FIELD HOMOGRAPHY ESTIMATION")
    print(f"  Images: {len(image_files)}")
    print("=" * 60)

    for idx, img_path in enumerate(image_files):
        print(f"\n{'#' * 60}")
        print(f"  IMAGE {idx+1}/{len(image_files)}: {img_path.name}")
        print(f"{'#' * 60}")

        img_output_dir = output_root / img_path.stem
        try:
            process_image(str(img_path), str(img_output_dir), use_gui=not args.cli)
        except Exception as e:
            print(f"  ERROR: {e}")

    print(f"\nAll done. Results in: {output_root}/")


if __name__ == "__main__":
    main()