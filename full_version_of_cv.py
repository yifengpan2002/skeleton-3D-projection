import cv2
import numpy as np
import argparse
from pathlib import Path

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
    image_files = sorted(
        file for file in input_path.iterdir()
        if file.suffix.lower() in IMAGE_EXTS
    )
else:
    raise FileNotFoundError(f"Not found: {input_path}")

print(f"Processing {len(image_files)} images...\n")

for index, img_path in enumerate(image_files):
    img_output_dir = output_root / img_path.stem
    img_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        # ============================================================
        # 1. Load image
        # ============================================================
        frame = cv2.imread(str(img_path))

        if frame is None:
            print(f"[SKIP] Could not read image: {img_path}")
            continue

        # ============================================================
        # 2. Green Field Masking
        # ============================================================
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 25, 25])
        upper_green = np.array([90, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Clean green mask
        green_kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, green_kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, green_kernel)

        # Keep only green field area
        green_field = cv2.bitwise_and(frame, frame, mask=green_mask)

        # ============================================================
        # 3. White Line Candidate Detection
        #    This is mainly for debug / contour visualization.
        # ============================================================
        gray_green = cv2.cvtColor(green_field, cv2.COLOR_BGR2GRAY)

        white_lines = cv2.adaptiveThreshold(
            gray_green,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            21,
            -5
        )

        # Restrict white candidates to green mask
        white_lines = cv2.bitwise_and(white_lines, white_lines, mask=green_mask)

        # Clean white-line mask
        line_kernel = np.ones((3, 3), np.uint8)
        white_lines_cleaned = cv2.morphologyEx(
            white_lines,
            cv2.MORPH_OPEN,
            line_kernel
        )

        connect_kernel = np.ones((5, 5), np.uint8)
        white_lines_cleaned = cv2.morphologyEx(
            white_lines_cleaned,
            cv2.MORPH_CLOSE,
            connect_kernel
        )

        # ============================================================
        # 4. Edge Detection
        #    Your idea:
        #    Use Canny(green_field), then remove Canny(green_mask)
        # ============================================================

        # Edge result from green field.
        # This is the best result according to your test.
        gray_green_field = cv2.cvtColor(green_field, cv2.COLOR_BGR2GRAY)
        edges_green_field = cv2.Canny(gray_green_field, 50, 150)

        # Edge result from green mask.
        # This usually represents artificial mask boundary / false positives.
        edges_green_mask = cv2.Canny(green_mask, 50, 150)

        # Dilate mask edges so the false positive removal is more complete.
        remove_kernel = np.ones((5, 5), np.uint8)
        edges_green_mask_dilated = cv2.dilate(
            edges_green_mask,
            remove_kernel,
            iterations=1
        )

        # Remove the mask-edge false positives from green-field edges.
        edges = cv2.bitwise_and(
            edges_green_field,
            cv2.bitwise_not(edges_green_mask_dilated)
        )

        # Optional: clean final edge result slightly
        small_kernel = np.ones((2, 2), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_OPEN, small_kernel)

        # ============================================================
        # 5. Contour Detection
        #    Use final edges or white_lines_cleaned depending on debug goal.
        # ============================================================
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contour_vis = frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)

            # Remove tiny noise
            if area < 30:
                continue

            cv2.drawContours(contour_vis, [cnt], -1, (0, 0, 255), 2)

        # ============================================================
        # 6. Geometric Line Extraction with Hough Transform
        # ============================================================
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=80,
            minLineLength=60,
            maxLineGap=25
        )

        line_vis = frame.copy()

        if lines is not None:
            print(f"[{img_path.name}] Detected {len(lines)} Hough line segments")

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
            print(f"[{img_path.name}] No Hough lines detected")

        # ============================================================
        # 7. Save debug outputs
        # ============================================================
        cv2.imwrite(str(img_output_dir / "01_original.jpg"), frame)
        cv2.imwrite(str(img_output_dir / "02_green_mask.jpg"), green_mask)
        cv2.imwrite(str(img_output_dir / "03_green_field.jpg"), green_field)

        cv2.imwrite(str(img_output_dir / "04_white_lines_threshold.jpg"), white_lines)
        cv2.imwrite(str(img_output_dir / "05_white_lines_cleaned.jpg"), white_lines_cleaned)

        cv2.imwrite(str(img_output_dir / "06_edges_green_field.jpg"), edges_green_field)
        cv2.imwrite(str(img_output_dir / "07_edges_green_mask.jpg"), edges_green_mask)
        cv2.imwrite(str(img_output_dir / "08_edges_green_mask_dilated.jpg"), edges_green_mask_dilated)
        cv2.imwrite(str(img_output_dir / "09_final_edges_removed_mask_edges.jpg"), edges)

        cv2.imwrite(str(img_output_dir / "10_contours_on_original.jpg"), contour_vis)
        cv2.imwrite(str(img_output_dir / "11_hough_lines_on_original.jpg"), line_vis)

        # ============================================================
        # 8. Optional display
        # ============================================================
        cv2.imshow("01 Original", frame)
        cv2.imshow("02 Green Mask", green_mask)
        cv2.imshow("03 Green Field", green_field)
        cv2.imshow("06 Edges Green Field", edges_green_field)
        cv2.imshow("07 Edges Green Mask", edges_green_mask)
        cv2.imshow("09 Final Edges", edges)
        cv2.imshow("11 Hough Lines", line_vis)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print(f"[OK] {index + 1}/{len(image_files)} processed: {img_path.name}")

    except Exception as e:
        print(f"[ERROR] Failed to process {img_path}: {e}")