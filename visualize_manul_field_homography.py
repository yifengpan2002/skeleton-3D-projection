import cv2
import numpy as np
import argparse
from pathlib import Path
# After computing H from field_homography.py:
H = np.load(r"outputs\homography\ultimate challenge but easier\homography.npy")
frame = cv2.imread(r"data\simple_rugby_field\ultimate challenge but easier.webp")

# Scale: 10 pixels per metre
SCALE = 10
FIELD_W = 100 * SCALE  # 1000px
FIELD_H = 70 * SCALE   # 700px

# Warp the original image to bird's-eye view
birds_eye = cv2.warpPerspective(frame, H, (FIELD_W, FIELD_H))

# Draw rugby field lines on top
def draw_rugby_template(img, scale=10):
    """Draw official rugby field lines on bird's-eye image."""
    s = scale
    color = (255, 255, 255)
    thick = 2

    # Touchlines (sidelines)
    cv2.line(img, (0, 0), (100*s, 0), color, thick)
    cv2.line(img, (0, 70*s), (100*s, 70*s), color, thick)

    # Try lines
    cv2.line(img, (0, 0), (0, 70*s), color, thick)
    cv2.line(img, (100*s, 0), (100*s, 70*s), color, thick)

    # 22m lines
    cv2.line(img, (22*s, 0), (22*s, 70*s), (255, 200, 0), thick)
    cv2.line(img, (78*s, 0), (78*s, 70*s), (255, 200, 0), thick)

    # 10m lines
    cv2.line(img, (40*s, 0), (40*s, 70*s), (200, 200, 200), 1)
    cv2.line(img, (60*s, 0), (60*s, 70*s), (200, 200, 200), 1)

    # Halfway line
    cv2.line(img, (50*s, 0), (50*s, 70*s), (0, 255, 0), thick)

    # Labels
    cv2.putText(img, "TRY", (2, 35*s), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    cv2.putText(img, "22m", (22*s+3, 35*s), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
    cv2.putText(img, "50m", (50*s+3, 35*s), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    cv2.putText(img, "22m", (78*s+3, 35*s), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,200,0), 1)
    cv2.putText(img, "TRY", (96*s, 35*s), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    return img

draw_rugby_template(birds_eye, SCALE)

cv2.imshow("Rugby Field - Bird's Eye View", birds_eye)
cv2.waitKey(0)
cv2.destroyAllWindows()