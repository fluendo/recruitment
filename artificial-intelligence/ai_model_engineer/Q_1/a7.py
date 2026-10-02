# Question 7 (extra): baseline for depth-based background blur.
# Idea: estimate depth with Depth Anything V2 Small, flood fill from each face that stays
# visible to find the person around it, keep that region sharp and blur everything else.

import cv2
import numpy as np
import torch
from depth_anything_v2.dpt import DepthAnythingV2  # from the official Depth-Anything-V2 repo

device = "cuda" if torch.cuda.is_available() else "cpu"


class DepthEstimator:
    def __init__(self, checkpoint="checkpoints/depth_anything_v2_vits.pth"):
        self.model = DepthAnythingV2(encoder="vits", features=64, out_channels=[48, 96, 192, 384])
        self.model.load_state_dict(torch.load(checkpoint, map_location=device))
        self.model.to(device).eval()
        self.model.requires_grad_(False)

    def __call__(self, img):
        return self.model.infer_image(img)  # H x W relative depth map (numpy)


def depth_blur(src, depth, fboxes, tol=0.03, feather=8):
    """Keep the faces in fboxes, and what touches them at similar depth, sharp. Blur the rest.

    tol is the depth tolerance (depth is scaled to 0-1).
    feather is the softness of the edge, in pixels.

    Baseline version: one seed pixel per face, no window or depth band.
    The fuller design is in Question 2.
    """
    h, w = src.shape[:2]
    blurred = cv2.blur(src, (50, 50))

    # relative depth has no fixed scale, so rescale it to 0-1 before using a tolerance.
    # Percentiles instead of min/max, so one outlier pixel doesn't change the scale.
    lo, hi = np.percentile(depth, (1, 99))
    d = np.clip((depth - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

    keep = np.zeros((h + 2, w + 2), np.uint8)  # floodFill needs a mask 2 px bigger than the image
    flags = 4 | cv2.FLOODFILL_MASK_ONLY | (255 << 8)  # only fill the mask, don't touch d
    for x1, y1, x2, y2 in fboxes:
        seed = ((x1 + x2) // 2, (y1 + y2) // 2)  # center of the face box, as (x, y)
        cv2.floodFill(d, keep, seed, 0, tol, tol, flags)

    keep = keep[1:-1, 1:-1] > 0  # back to the image size

    # soft edge: blur the mask and blend, instead of a hard cut
    alpha = cv2.GaussianBlur(keep.astype(np.float32), (0, 0), feather)[..., None]
    out = alpha * src + (1 - alpha) * blurred
    return out.astype(np.uint8)


def main():
    img = cv2.imread("path/to/img.jpg")
    # faces that stay visible, as (x1, y1, x2, y2), from the detector + allow-list (Question 1)
    fboxes = [(120, 80, 220, 200)]

    estimator = DepthEstimator()
    depth = estimator(img)
    out = depth_blur(img, depth, fboxes)
    cv2.imwrite("out.jpg", out)


if __name__ == "__main__":
    main()
