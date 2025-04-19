"""Tests for geometry module.

This module tests the geometric functions for multi-view geometry,
using known camera poses and ideal correspondences.
"""

import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import geometry


class TestGeometry(unittest.TestCase):
    """Test geometric functions."""

    def setUp(self):
        """Set up test data with known camera poses and 3D points."""
        # Define camera intrinsics
        self.K = np.array([
            [800, 0, 400],
            [0, 800, 300],
            [0, 0, 1]
        ])
        
        # Define 3D points
        self.points3d = np.array([
            [0, 0, 4],
            [1, 0, 4],
            [0, 1, 4],
            [-1, 0, 4],
            [0, -1, 4],
            [1, 1, 4],
            [-1, 1, 4],
            [1, -1, 4],
            [-1, -1, 4],
            [0.5, 0, 3]
        ])
        
        # Define camera poses for two views
        # First camera at origin
        self.R1 = np.eye(3)
        self.t1 = np.zeros(3)
        self.P1 = np.hstack((self.R1, self.t1.reshape(3, 1)))
        
        # Second camera translated
        self.R2 = np.array([
            [0.866, 0, -0.5],
            [0, 1, 0],
            [0.5, 0, 0.866]
        ])  # Rotation around Y axis by 30 degrees
        self.t2 = np.array([-1, 0, 0])  # Translation along X axis
        self.P2 = np.hstack((self.R2, self.t2.reshape(3, 1)))
        
        # Camera matrices
        self.K1 = self.K
        self.K2 = self.K
        
        # Projection matrices
        self.proj1 = self.K1 @ self.P1
        self.proj2 = self.K2 @ self.P2
        
        # Generate perfect matches by projecting 3D points
        self.matches = []
        for point in self.points3d:
            # Project to first view
            p1 = self.proj1 @ np.append(point, 1)
            p1 = p1[:2] / p1[2]
            
            # Project to second view
            p2 = self.proj2 @ np.append(point, 1)
            p2 = p2[:2] / p2[2]
            
            # Store match
            self.matches.append(np.hstack((p1, p2)))
        
        self.matches = np.array(self.matches)

    def test_normalize_points(self):
        """Test point normalization."""
        # Take the first view's projected points
        points = self.matches[:, :2]
        
        # Normalize them
        normalized, T = geometry.normalize_points(points)
        
        # Check mean is close to zero
        mean = np.mean(normalized, axis=0)
        self.assertAlmostEqual(mean[0], 0, delta=1e-12)
        self.assertAlmostEqual(mean[1], 0, delta=1e-12)
        
        # Check average distance from origin is sqrt(2)
        distances = np.sqrt(np.sum(normalized**2, axis=1))
        avg_distance = np.mean(distances)
        self.assertAlmostEqual(avg_distance, np.sqrt(2), delta=1e-12)
        
        # Check that applying T to original points gives normalized points
        points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed = (T @ points_homogeneous.T).T[:, :2]
        np.testing.assert_allclose(normalized, transformed, rtol=1e-12, atol=1e-12)

    def test_fundamental_matrix(self):
        """Test fundamental matrix estimation from matches."""
        # Estimate fundamental matrix
        F = geometry.fundamental_8pt(self.matches)
        
        # Properties of a fundamental matrix
        # 1. Rank 2
        rank = np.linalg.matrix_rank(F, tol=1e-10)
        self.assertEqual(rank, 2, "Fundamental matrix should have rank 2")
        
        # 2. Determinant close to zero
        det = np.linalg.det(F)
        self.assertAlmostEqual(det, 0, delta=1e-10)
        
        # 3. Epipolar constraint: x2.T @ F @ x1 = 0
        for match in self.matches:
            x1 = np.append(match[:2], 1)  # Homogeneous coords
            x2 = np.append(match[2:], 1)
            
            # Compute epipolar constraint
            epipolar_constraint = x2 @ F @ x1
            
            # Should be close to zero
            self.assertAlmostEqual(epipolar_constraint, 0, delta=1e-8)
        
        # 4. Consistency with epipolar geometry
        # Compute epipolar lines
        for match in self.matches:
            x1 = np.append(match[:2], 1)  # Homogeneous coords
            x2 = np.append(match[2:], 1)
            
            # Compute epipolar line in second image (l2 = F @ x1)
            line = F @ x1
            
            # Point x2 should lie on this line: x2.T @ line = 0
            point_on_line = x2 @ line
            self.assertAlmostEqual(point_on_line, 0, delta=1e-8)

    def test_essential_matrix(self):
        """Test essential matrix computation from fundamental matrix."""
        # First compute fundamental matrix
        F = geometry.fundamental_8pt(self.matches)
        
        # Compute essential matrix
        E = geometry.essential_from_F(F, self.K1, self.K2)
        
        # Properties of an essential matrix
        # 1. Has rank 2
        rank = np.linalg.matrix_rank(E, tol=1e-10)
        self.assertEqual(rank, 2, "Essential matrix should have rank 2")
        
        # 2. Two singular values should be equal, third should be zero
        u, s, vh = np.linalg.svd(E)
        self.assertAlmostEqual(s[0], s[1], delta=1e-10)
        self.assertAlmostEqual(s[2], 0, delta=1e-10)

    def test_decompose_E(self):
        """Test decomposition of essential matrix into R and t."""
        # First compute fundamental and essential matrices
        F = geometry.fundamental_8pt(self.matches)
        E = geometry.essential_from_F(F, self.K1, self.K2)
        
        # Decompose E
        poses = geometry.decompose_E(E)
        
        # Should have 4 possible camera poses
        self.assertEqual(len(poses), 4, "Should have 4 possible poses")
        
        # At least one should be similar to our ground truth R2, t2
        found_match = False
        for R, t in poses:
            # Check if R is similar to ground truth R2
            # Note: There might be ambiguity in the sign of t
            if (np.allclose(R, self.R2, atol=1e-2) and 
                (np.allclose(t, self.t2.reshape(3, 1), atol=1e-2) or
                 np.allclose(t, -self.t2.reshape(3, 1), atol=1e-2))):
                found_match = True
                break
        
        self.assertTrue(found_match, "One pose should match ground truth")

    def test_triangulate_point(self):
        """Test point triangulation."""
        # Use the first match for triangulation
        match = self.matches[0]
        
        # Triangulate
        X = geometry.triangulate_point(self.proj1, self.proj2, match[:2], match[2:])
        
        # Convert to non-homogeneous
        X_nonhomogeneous = X[:3] / X[3]
        
        # Should be close to the original 3D point
        np.testing.assert_allclose(X_nonhomogeneous, self.points3d[0], atol=1e-6)

    def test_triangulate_all_points(self):
        """Test triangulation of all points."""
        triangulated_points = []
        
        for i, match in enumerate(self.matches):
            # Triangulate
            X = geometry.triangulate_point(self.proj1, self.proj2, match[:2], match[2:])
            
            # Convert to non-homogeneous
            X_nonhomogeneous = X[:3] / X[3]
            triangulated_points.append(X_nonhomogeneous)
        
        triangulated_points = np.array(triangulated_points)
        
        # All triangulated points should be close to the original 3D points
        np.testing.assert_allclose(triangulated_points, self.points3d, atol=1e-6)

    def test_find_best_pose(self):
        """Test finding the best pose using cheirality check."""
        # First compute fundamental and essential matrices
        F = geometry.fundamental_8pt(self.matches)
        E = geometry.essential_from_F(F, self.K1, self.K2)
        
        # Decompose E to get pose candidates
        poses = geometry.decompose_E(E)
        
        # Find best pose
        R, t, points3d = geometry.find_best_pose(poses, self.K1, self.K2, self.matches)
        
        # Check that R, t are close to ground truth
        np.testing.assert_allclose(R, self.R2, atol=1e-2)
        
        # t might be scaled, check the direction
        t_normalized = t / np.linalg.norm(t)
        t2_normalized = self.t2.reshape(3, 1) / np.linalg.norm(self.t2)
        np.testing.assert_allclose(t_normalized, t2_normalized, atol=1e-2)
        
        # Calculate reprojection error
        P1 = self.K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = self.K2 @ np.hstack((R, t))
        
        errors = []
        for i, point in enumerate(points3d):
            # Project 3D point to both views
            p1 = P1 @ np.append(point[:3], 1)
            p1 = p1[:2] / p1[2]
            
            p2 = P2 @ np.append(point[:3], 1)
            p2 = p2[:2] / p2[2]
            
            # Compute reprojection error
            error1 = np.linalg.norm(p1 - self.matches[i, :2])
            error2 = np.linalg.norm(p2 - self.matches[i, 2:])
            errors.append(error1 + error2)
        
        # Average reprojection error should be small
        mean_error = np.mean(errors)
        self.assertLess(mean_error, 1e-6, f"Reprojection error too high: {mean_error}")

    def test_pnp_ransac(self):
        """Test PnP RANSAC."""
        # Use 3D points and their projections in second view
        points3d = self.points3d
        points2d = self.matches[:, 2:4]
        
        # Run PnP
        pose = geometry.pnp_ransac(points3d, points2d, self.K2)
        
        # Extract R and t
        R_estimated = pose[:, :3]
        t_estimated = pose[:, 3]
        
        # Check that R is close to ground truth
        np.testing.assert_allclose(R_estimated, self.R2, atol=1e-2)
        
        # Check that t is close to ground truth (might be scaled)
        t_normalized = t_estimated / np.linalg.norm(t_estimated)
        t2_normalized = self.t2 / np.linalg.norm(self.t2)
        np.testing.assert_allclose(t_normalized, t2_normalized, atol=1e-2)
        
        # Compute reprojection error
        P = self.K2 @ pose
        
        errors = []
        for i, point in enumerate(points3d):
            # Project 3D point
            p = P @ np.append(point, 1)
            p = p[:2] / p[2]
            
            # Compute reprojection error
            error = np.linalg.norm(p - points2d[i])
            errors.append(error)
        
        # Average reprojection error should be small
        mean_error = np.mean(errors)
        self.assertLess(mean_error, 1e-6, f"Reprojection error too high: {mean_error}")


if __name__ == "__main__":
    unittest.main()
