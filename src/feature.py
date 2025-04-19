"""Feature detection and matching module.

This module implements feature detection and matching using SIFT or ORB
algorithms with FLANN matching and Lowe's ratio test for filtering.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def detect_and_match(
    images: list[np.ndarray], method: str = "sift", ratio: float = 0.75
) -> dict[tuple[int, int], np.ndarray]:
    """Detect features and match across multiple images.

    Detect SIFT (default) or ORB keypoints and perform matching with FLANN
    and Lowe's ratio test filtering. Returns matched point pairs for each
    image pair with sufficient matches.

    Args:
        images: List of input images as numpy arrays
        method: Feature detection method ("sift" or "orb")
        ratio: Lowe's ratio test threshold

    Returns:
        Dictionary mapping image pairs (i,j) to Nx4 arrays of point
        correspondences [x1,y1,x2,y2]
    """
    start_time = time.perf_counter()
    n_images = len(images)
    logger.info(f"Detecting features using {method.upper()} on {n_images} images")

    # Initialize detector based on method
    if method.lower() == "sift":
        detector = cv2.SIFT_create(
            nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10
        )
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
    elif method.lower() == "orb":
        detector = cv2.ORB_create(
            nfeatures=2000, scaleFactor=1.2, nlevels=8, fastThreshold=20
        )
        # FLANN parameters for ORB (using LSH)
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1,
        )
        search_params = dict(checks=50)
    else:
        raise ValueError(f"Unknown feature detection method: {method}")

    # Detect keypoints and compute descriptors
    keypoints = []
    descriptors = []
    for i, img in enumerate(tqdm(images, desc="Detecting features")):
        if len(img.shape) == 3 and img.shape[2] == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        kp, des = detector.detectAndCompute(gray, None)
        keypoints.append(kp)
        descriptors.append(des)
        logger.debug(f"Image {i}: detected {len(kp)} keypoints")

    # Match features between all image pairs
    matches_dict = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    total_pairs = (n_images * (n_images - 1)) // 2
    with tqdm(total=total_pairs, desc="Matching features") as pbar:
        for i in range(n_images):
            for j in range(i + 1, n_images):
                # Skip if either image has no descriptors
                if descriptors[i] is None or descriptors[j] is None:
                    pbar.update(1)
                    continue

                # Convert descriptors to appropriate type
                if method.lower() == "orb":
                    des1 = descriptors[i].astype(np.uint8)
                    des2 = descriptors[j].astype(np.uint8)
                else:
                    des1 = descriptors[i]
                    des2 = descriptors[j]

                # Match descriptors using FLANN
                if des1.shape[0] > 0 and des2.shape[0] > 0:
                    matches = flann.knnMatch(des1, des2, k=2)
                    
                    # Apply Lowe's ratio test
                    good_matches = []
                    for m, n in matches:
                        if m.distance < ratio * n.distance:
                            good_matches.append(m)
                    
                    if len(good_matches) >= 8:  # minimum needed for F matrix
                        # Extract coordinates of matched keypoints
                        pts1 = np.float32([keypoints[i][m.queryIdx].pt for m in good_matches])
                        pts2 = np.float32([keypoints[j][m.trainIdx].pt for m in good_matches])
                        
                        # Store as [x1, y1, x2, y2] array
                        match_array = np.hstack((pts1, pts2))
                        matches_dict[(i, j)] = match_array
                        
                        logger.debug(
                            f"Image pair ({i},{j}): {len(good_matches)} matches after ratio test"
                        )

                pbar.update(1)

    elapsed_time = time.perf_counter() - start_time
    logger.info(
        f"Feature matching complete: {len(matches_dict)} image pairs with matches"
        f" (elapsed time: {elapsed_time:.2f}s)"
    )
    
    return matches_dict


def display_matches(
    img1: np.ndarray, img2: np.ndarray, match_points: np.ndarray, n_matches: int = 50
) -> np.ndarray:
    """Display matched features between two images.

    Args:
        img1: First image
        img2: Second image
        match_points: Nx4 array of point correspondences [x1,y1,x2,y2]
        n_matches: Number of matches to display (randomly sampled)

    Returns:
        Visualization image with matches drawn between image pairs
    """
    # Convert images to grayscale if needed
    if len(img1.shape) == 3:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1

    if len(img2.shape) == 3:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2

    # Convert grayscale to color for drawing
    img1_display = cv2.cvtColor(img1_gray, cv2.COLOR_GRAY2BGR)
    img2_display = cv2.cvtColor(img2_gray, cv2.COLOR_GRAY2BGR)

    # Create a new image that contains both images side by side
    h1, w1 = img1_display.shape[:2]
    h2, w2 = img2_display.shape[:2]
    h = max(h1, h2)
    w = w1 + w2
    vis = np.zeros((h, w, 3), dtype=np.uint8)
    vis[:h1, :w1] = img1_display
    vis[:h2, w1:w1+w2] = img2_display

    # Select a subset of matches to display
    n_total = match_points.shape[0]
    if n_total > n_matches:
        indices = np.random.choice(n_total, n_matches, replace=False)
        subset = match_points[indices]
    else:
        subset = match_points

    # Draw lines between matches
    for i in range(subset.shape[0]):
        pt1 = (int(subset[i, 0]), int(subset[i, 1]))
        pt2 = (int(subset[i, 2]) + w1, int(subset[i, 3]))
        
        # Random color for each match
        color = np.random.randint(0, 255, 3).tolist()
        
        cv2.line(vis, pt1, pt2, color, 1)
        cv2.circle(vis, pt1, 3, color, -1)
        cv2.circle(vis, pt2, 3, color, -1)

    return vis
