"""Tests for feature detection and matching module.

This module tests the feature detection and matching functionality,
ensuring we can detect and match features between images.
"""

import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import feature


class TestFeatureDetection(unittest.TestCase):
    """Test feature detection and matching."""

    def setUp(self):
        """Set up test data."""
        # Create a simple synthetic test image pair with a known transformation
        self.img1 = np.zeros((300, 400), dtype=np.uint8)
        
        # Add some corners and patterns to make good features
        self.img1[50:100, 50:100] = 255  # Square
        self.img1[150:170, 150:250] = 255  # Rectangle
        self.img1[200:250, 200:250] = 255  # Square
        
        # Create the second image with a translation
        tx, ty = 20, 10  # Translation
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        self.img2 = cv2.warpAffine(self.img1, M, (400, 300))
        
        # Store the ground truth transformation for reference
        self.tx = tx
        self.ty = ty

    def test_sift_detection(self):
        """Test SIFT feature detection."""
        # Detect SIFT features in both images
        sift = cv2.SIFT_create()
        kp1, des1 = sift.detectAndCompute(self.img1, None)
        kp2, des2 = sift.detectAndCompute(self.img2, None)
        
        # Verify we found some keypoints
        self.assertGreater(len(kp1), 0, "Should detect keypoints in first image")
        self.assertGreater(len(kp2), 0, "Should detect keypoints in second image")

    def test_detect_and_match(self):
        """Test feature detection and matching between two images."""
        # Convert grayscale to color for testing
        img1_color = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)
        
        # Run our detect_and_match function
        images = [img1_color, img2_color]
        matches = feature.detect_and_match(images, method="sift", ratio=0.75)
        
        # Check we have matches between image pair (0,1)
        self.assertIn((0, 1), matches, "Should find matches between images")
        
        # Check we have enough matches (at least 30)
        match_points = matches[(0, 1)]
        self.assertGreaterEqual(
            len(match_points), 30, 
            f"Expected at least 30 matches, got {len(match_points)}"
        )
        
        # Check match format is correct
        self.assertEqual(
            match_points.shape[1], 4,
            "Match points should be Nx4 array [x1,y1,x2,y2]"
        )
        
        # Verify the translation in matches
        # Since it's synthetic, most matches should have approximately our known translation
        diffs_x = match_points[:, 2] - match_points[:, 0]  # x2 - x1
        diffs_y = match_points[:, 3] - match_points[:, 1]  # y2 - y1
        
        mean_diff_x = np.mean(diffs_x)
        mean_diff_y = np.mean(diffs_y)
        
        # Allow some tolerance due to feature localization precision
        self.assertAlmostEqual(
            mean_diff_x, self.tx, delta=5,
            msg=f"Expected x translation {self.tx}, got {mean_diff_x}"
        )
        self.assertAlmostEqual(
            mean_diff_y, self.ty, delta=5,
            msg=f"Expected y translation {self.ty}, got {mean_diff_y}"
        )

    def test_orb_matching(self):
        """Test ORB feature detection and matching."""
        # Convert grayscale to color for testing
        img1_color = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)
        
        # Run our detect_and_match function with ORB
        images = [img1_color, img2_color]
        matches = feature.detect_and_match(images, method="orb", ratio=0.75)
        
        # Check we have matches between image pair (0,1)
        self.assertIn((0, 1), matches, "Should find matches between images")
        
        # Check we have enough matches
        match_points = matches[(0, 1)]
        self.assertGreaterEqual(
            len(match_points), 10,  # ORB might give fewer good matches than SIFT
            f"Expected at least 10 matches, got {len(match_points)}"
        )

    def test_display_matches(self):
        """Test visualization of matches."""
        # Convert grayscale to color for testing
        img1_color = cv2.cvtColor(self.img1, cv2.COLOR_GRAY2BGR)
        img2_color = cv2.cvtColor(self.img2, cv2.COLOR_GRAY2BGR)
        
        # Run our detect_and_match function
        images = [img1_color, img2_color]
        matches = feature.detect_and_match(images, method="sift", ratio=0.75)
        
        # Create visualization
        match_points = matches[(0, 1)]
        vis = feature.display_matches(img1_color, img2_color, match_points)
        
        # Check the visualization has the right shape
        expected_width = img1_color.shape[1] + img2_color.shape[1]
        expected_height = max(img1_color.shape[0], img2_color.shape[0])
        
        self.assertEqual(
            vis.shape[:2], (expected_height, expected_width),
            "Visualization has incorrect dimensions"
        )


if __name__ == "__main__":
    unittest.main()
