"""
detect_field_lines.py
----------------------
Rugby field line detection using color segmentation + Hough transform.

Usage:
    python detect_field_lines.py -i "data/simple_rugby_field"
    python detect_field_lines.py -i "data/simple_rugby_field/image.jpg"
"""

import cv2
import numpy as np
import argparse
from pathlib import Path


def find_field_mask(hsv, img_shape):
    """Find the largest green region (the field)."""
    green_mask = cv2.inRange(hsv, np.array([30, 30, 30]), np.array([90, 255, 255]))
    kernel = np.ones((25, 25), np.uint8)
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return green_mask

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    field_mask = np.zeros(img_shape[:2], dtype=np.uint8)
    cv2.drawContours(field_mask, [contours[0]], -1, 255, cv2.FILLED)
    return cv2.morphologyEx(field_mask, cv2.MORPH_CLOSE, kernel)


def detect_white_on_field(img, hsv, field_mask):
    """Detect white line pixels on the field."""
    white_hsv = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 60, 255]))

    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    white_adaptive = cv2.adaptiveThreshold(
        hls[:, :, 1], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, -15
    )

    white_mask = cv2.bitwise_or(white_hsv, white_adaptive)
    white_mask = cv2.bitwise_and(white_mask, field_mask)

    k = np.ones((3, 3), np.uint8)
    white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, k)
    return cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, k)


def detect_field_lines(image_path, output_dir="outputs/field_lines"):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 1. Field mask
    field_mask = find_field_mask(hsv, img.shape)

    # 2. White lines on field
    white_mask = detect_white_on_field(img, hsv, field_mask)

    # 3. Edge detection + Hough lines (directly on white mask)
    edges = cv2.Canny(white_mask, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180,
        threshold=30, minLineLength=40, maxLineGap=30,
    )

    line_count = len(lines) if lines is not None else 0

    # 4. Draw result
    result = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite(f"{output_dir}/result.jpg", result)
    cv2.imwrite(f"{output_dir}/white_mask.jpg", white_mask)

    print(f"    {Path(image_path).name}: {line_count} lines -> {output_dir}/result.jpg")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True)
    parser.add_argument("--output", "-o", type=str, default="outputs/field_lines")
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

    print(f"Processing {len(image_files)} image(s)...\n")
    for idx, img_path in enumerate(image_files):
        img_output_dir = output_root / img_path.stem
        try:
            detect_field_lines(str(img_path), str(img_output_dir))
        except Exception as e:
            print(f"    {img_path.name}: ERROR - {e}")

    print(f"\nDone. Results in: {output_root}/")


if __name__ == "__main__":
    main()