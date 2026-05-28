import cv2
import numpy as np
import argparse
from pathlib import Path

import tkinter as tk
from PIL import Image, ImageTk

def show_fit(name, img):
    """Display image in a scrollable window (blocks until closed)."""
    # Convert OpenCV BGR to PIL RGB
    if len(img.shape) == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img  # grayscale
    pil_img = Image.fromarray(img_rgb)

    root = tk.Tk()
    root.title(name)
    root.geometry("1280x720")

    frame = tk.Frame(root)
    frame.pack(fill="both", expand=True)

    canvas = tk.Canvas(frame, bg="black")
    h_scroll = tk.Scrollbar(frame, orient="horizontal", command=canvas.xview)
    v_scroll = tk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

    h_scroll.pack(side="bottom", fill="x")
    v_scroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)

    photo = ImageTk.PhotoImage(pil_img)
    canvas.create_image(0, 0, anchor="nw", image=photo)
    canvas.image = photo
    canvas.configure(scrollregion=canvas.bbox("all"))

    root.mainloop()   # ← blocks until you close the window
def find_field_polygon(lines):
    """
    Find field boundary using longest lines as anchors.
    
    Logic:
    1. Sort lines by length
    2. Longest line = trusted reference (e.g., touchline)
    3. Find longest perpendicular line
    4. Find their parallels (2nd touchline, 2nd try line)
    5. Four intersections = four field corners
    """
    if lines is None or len(lines) < 3:
        return None

    # Calculate length and angle for each line
    line_info = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
        angle = np.degrees(np.arctan2(y2-y1, x2-x1)) % 180
        line_info.append({
            'pts': (x1, y1, x2, y2),
            'length': length,
            'angle': angle,
        })

    # Sort by length (longest first)
    line_info.sort(key=lambda x: x['length'], reverse=True)

    # Longest line = reference
    ref = line_info[0]
    ref_angle = ref['angle']

    # Split remaining lines into PARALLEL and PERPENDICULAR to reference
    parallel = []     # Same direction as reference (other touchline)
    perpendicular = []  # ~90° to reference (try lines)

    for li in line_info[1:]:
        angle_diff = abs(li['angle'] - ref_angle)
        if angle_diff > 90:
            angle_diff = 180 - angle_diff

        if angle_diff < 20:
            parallel.append(li)
        elif angle_diff > 60:
            perpendicular.append(li)

    # Pick the best parallel (longest, farthest from reference)
    ref_mid_y = (ref['pts'][1] + ref['pts'][3]) / 2
    best_parallel = None
    best_dist = 0
    for li in parallel:
        mid_y = (li['pts'][1] + li['pts'][3]) / 2
        dist = abs(mid_y - ref_mid_y)
        if dist > best_dist:
            best_dist = dist
            best_parallel = li

    if best_parallel is None or not perpendicular:
        return None

    # Pick longest perpendicular as one try line
    perp1 = perpendicular[0]

    # Find second perpendicular (farthest from first)
    perp1_mid_x = (perp1['pts'][0] + perp1['pts'][2]) / 2
    best_perp2 = None
    best_perp_dist = 0
    for li in perpendicular[1:]:
        mid_x = (li['pts'][0] + li['pts'][2]) / 2
        dist = abs(mid_x - perp1_mid_x)
        if dist > best_perp_dist:
            best_perp_dist = dist
            best_perp2 = li

    # We have 2-4 boundary lines, find their intersections
    def line_intersect(l1, l2):
        x1, y1, x2, y2 = l1['pts']
        x3, y3, x4, y4 = l2['pts']
        denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
        if abs(denom) < 1e-6:
            return None
        t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
        px = x1 + t*(x2-x1)
        py = y1 + t*(y2-y1)
        return (int(px), int(py))

    # Build the 4 corners from line intersections
    boundary_lines = [ref, best_parallel]
    if best_perp2:
        perp_lines = [perp1, best_perp2]
    else:
        # Only 3 lines: extend perpendicular to both sides
        perp_lines = [perp1]

    corners = []
    for h_line in boundary_lines:
        for v_line in perp_lines:
            pt = line_intersect(h_line, v_line)
            if pt:
                corners.append(pt)

    if len(corners) < 3:
        return None

    # Order corners clockwise
    corners = np.array(corners)
    center = corners.mean(axis=0)
    angles = np.arctan2(corners[:, 1] - center[1], corners[:, 0] - center[0])
    corners = corners[np.argsort(angles)]

    return corners

parser = argparse.ArgumentParser()
parser.add_argument("--input", "-i", type=str, required=True)
parser.add_argument("--output", "-o", type=str, default="output/cv_filter")
arg = parser.parse_args()

input_path = Path(arg.input)
output_root = Path(arg.output)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}

if input_path.is_file():
    image_files = [input_path]
elif input_path.is_dir():
    image_files = sorted(file for file in input_path.iterdir() if file.suffix.lower() in IMAGE_EXTS)
else:
    raise FileNotFoundError(f"Not found; {input_path}")

print(f"Processing {len(image_files)} images... \n")

for index, img_path in enumerate(image_files):
    img_output_dir = output_root / img_path.stem
    try:
        #start the processing here

        #step1: isloating the green grass
        frame = cv2.imread(img_path)
        #change rbg color to hsv
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # define the range of green pixel
        lower_green = np.array([35, 25, 25])
        upper_green = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # denoise processing
        kernel = np.ones((5,5),  np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # === ADD THIS: Keep only the largest green region (the field) ===
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(green_mask, connectivity=8)
        if num_labels > 1:
            # Find largest component (skip background = label 0)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            green_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        # ================================================================

        # Fill any holes inside the field after isolating it
        big_kernel = np.ones((1, 1), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, big_kernel)

        green_field = cv2.bitwise_and(frame, frame, mask=green_mask)
        # cv2.imshow("Original Frame", frame)
        # cv2.imshow("Green Mask", green_mask)
        # cv2.imshow("Green Field", green_field)

        # functiton 2, Edge & Contour Detection start here
        gray_green = cv2.cvtColor(green_field, cv2.COLOR_BGR2GRAY)
        white_lines = cv2.adaptiveThreshold(
            gray_green,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -5
        )
        # 11. Clean small noise
        line_kernel = np.ones((3, 3), np.uint8)
        white_lines_cleaned = cv2.morphologyEx(white_lines, cv2.MORPH_OPEN, line_kernel)

        # Optional: connect broken line segments slightly
        connect_kernel = np.ones((5, 5), np.uint8)
        white_lines_cleaned = cv2.morphologyEx(white_lines_cleaned, cv2.MORPH_CLOSE, connect_kernel)

        # 12. Edge detection on cleaned white line mask
        # Edges from green_field (has both true + false positives)
        edges_all = cv2.Canny(green_field, 50, 150)

        # Edges from green_mask (has only false positives — the mask boundary)
        edges_boundary = cv2.Canny(green_mask, 50, 150)

        # Dilate boundary edges slightly to ensure full removal
        boundary_kernel = np.ones((5, 5), np.uint8)
        edges_boundary_dilated = cv2.dilate(edges_boundary, boundary_kernel, iterations=1)

        # Subtract: keep true positives, remove false positives
        edges = cv2.subtract(edges_all, edges_boundary_dilated)

        # 13. Find contours of line markings
        contours, _ = cv2.findContours(
            white_lines_cleaned,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # 14. Draw contours on original frame
        contour_vis = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Remove tiny noise
            if area < 50:
                continue

            cv2.drawContours(contour_vis, [cnt], -1, (0, 0, 255), 2)

        # 15. Save outputs
        show_fit(str(img_output_dir / "04_white_lines_threshold.jpg"), white_lines)
        show_fit(str(img_output_dir / "05_white_lines_cleaned.jpg"), white_lines_cleaned)
        show_fit(str(img_output_dir / "06_edges.jpg"), edges)
        show_fit(str(img_output_dir / "07_contours_on_original.jpg"), contour_vis)

        # function 3: Geometric Line Extraction using Hough Transform

        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=10,
            maxLineGap=10
        )

        # find the rectangular shape

        polygon = find_field_polygon(lines)

        vis = frame.copy()
        if polygon is not None:
            # Draw field boundary
            cv2.polylines(vis, [polygon], True, (0, 255, 0), 3)
            # Draw corners
            for i, pt in enumerate(polygon):
                cv2.circle(vis, tuple(pt), 10, (0, 0, 255), -1)
                cv2.putText(vis, f"C{i}", (pt[0]+15, pt[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        show_fit("Field Boundary", vis)

        line_vis = frame.copy()

        if lines is not None:
            print(f"Detected {len(lines)} Hough line segments")

            for line in lines:
                x1, y1, x2, y2 = line[0]

                cv2.line(
                    line_vis,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )
        else:
            print("No Hough lines detected")
        
        # text = f"Hough Lines: {line_count}"
        # cv2.putText(line_vis, text, (20, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)   # 黑色描边
        # cv2.putText(line_vis, text, (20, 50),
        #             cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)  # 黄色文字

        show_fit(str(img_output_dir / "08_hough_lines_on_original.jpg"), line_vis)

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    except Exception as e:
        print(f"[ERROR] Failed to process {img_path}: {e}")





# # 1. Load video frame
# frame = cv2.imread(r'C:\Users\Yifeng Pan\Documents\compsci760\skeleton-3D-projection\data\simple_rugby_field\partial img.jpg')
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# # 2. Isolate white lines (Adaptive Thresholding to combat stadium lighting)
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
#                                cv2.THRESH_BINARY, 15, -2)

# # 3. Clean up noise using Morphological Operations
# kernel = np.ones((3,3), np.uint8)
# cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

# # 4. Detect lines using Probabilistic Hough Transform
# lines = cv2.HoughLinesP(cleaned, rho=1, theta=np.pi/180, threshold=100, 
#                         minLineLength=80, maxLineGap=20)

# # 5. Draw lines back onto original frame
# if lines is not None:
#     for line in lines:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.imshow('Detected Rugby Lines', frame)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
