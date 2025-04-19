"""Geometric functions for multi-view geometry.

This module implements core geometric algorithms for multi-view geometry,
including point normalization, fundamental/essential matrix estimation,
camera pose recovery, and triangulation.
"""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

import cv2
import numpy as np
from scipy import linalg

logger = logging.getLogger(__name__)


def normalize_points(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Normalize points using Hartley's method.

    Applies isotropic scaling to 2D points so they have zero mean and
    average distance of sqrt(2) from the origin.

    Args:
        pts: Nx2 array of 2D points

    Returns:
        Tuple of (normalized_points, transformation_matrix)
    """
    if pts.shape[1] != 2:
        raise ValueError(f"Expected Nx2 points array, got shape {pts.shape}")

    # Calculate centroid
    centroid = np.mean(pts, axis=0)
    
    # Center the points
    centered_pts = pts - centroid
    
    # Calculate average distance from origin
    distances = np.sqrt(np.sum(centered_pts**2, axis=1))
    avg_distance = np.mean(distances)
    
    # Scale factor to achieve sqrt(2) average distance
    scale = np.sqrt(2) / avg_distance if avg_distance > 0 else 1.0
    
    # Create transformation matrix
    T = np.array([
        [scale, 0, -scale * centroid[0]],
        [0, scale, -scale * centroid[1]],
        [0, 0, 1]
    ])
    
    # Apply transformation to points
    pts_homogeneous = np.hstack((pts, np.ones((pts.shape[0], 1))))
    normalized_pts = (T @ pts_homogeneous.T).T[:, :2]
    
    logger.debug(f"Normalized {pts.shape[0]} points: scale={scale:.5f}")
    return normalized_pts, T


def fundamental_8pt(xy_pairs: np.ndarray) -> np.ndarray:
    """Estimate fundamental matrix using the 8-point algorithm.

    Estimates the fundamental matrix from point correspondences
    using the normalized 8-point algorithm, and enforces the
    rank-2 constraint through SVD.

    Args:
        xy_pairs: Nx4 array of point correspondences [x1,y1,x2,y2]

    Returns:
        3x3 fundamental matrix
    """
    if xy_pairs.shape[0] < 8:
        raise ValueError(f"At least 8 point pairs required, got {xy_pairs.shape[0]}")
    
    # Extract point coordinates
    pts1 = xy_pairs[:, :2]
    pts2 = xy_pairs[:, 2:]
    
    # Normalize points
    norm_pts1, T1 = normalize_points(pts1)
    norm_pts2, T2 = normalize_points(pts2)
    
    # Build the design matrix A for the homogeneous system
    n_points = norm_pts1.shape[0]
    A = np.zeros((n_points, 9))
    
    # Fill design matrix: [x'x, x'y, x', y'x, y'y, y', x, y, 1]
    for i in range(n_points):
        x1, y1 = norm_pts1[i]
        x2, y2 = norm_pts2[i]
        A[i] = [x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1]
    
    # Compute the SVD of A
    U, S, Vt = np.linalg.svd(A)
    
    # The solution is the last column of V (last row of Vt)
    F = Vt[-1].reshape(3, 3)
    
    # Enforce rank-2 constraint
    U_f, S_f, Vt_f = np.linalg.svd(F)
    # Set the smallest singular value to 0
    S_f[2] = 0
    # Reconstruct F with rank 2
    F_rank2 = U_f @ np.diag(S_f) @ Vt_f
    
    # Denormalize: F = T2.T @ F @ T1
    F_denorm = T2.T @ F_rank2 @ T1
    
    # Scale F so that ||F|| = 1 (Frobenius norm)
    F_denorm = F_denorm / np.linalg.norm(F_denorm)
    
    # Log matrix properties
    u, s, vt = np.linalg.svd(F_denorm)
    logger.debug(
        f"Fundamental matrix: shape={F_denorm.shape}, "
        f"rank={np.linalg.matrix_rank(F_denorm)}, "
        f"condition={s[0]/s[1]:.2f}, "
        f"singular values=[{s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f}]"
    )
    
    return F_denorm


def essential_from_F(F: np.ndarray, K1: np.ndarray, K2: np.ndarray) -> np.ndarray:
    """Convert fundamental matrix to essential matrix.

    E = K2.T @ F @ K1

    Args:
        F: 3x3 Fundamental matrix
        K1: 3x3 Intrinsic matrix for the first camera
        K2: 3x3 Intrinsic matrix for the second camera

    Returns:
        3x3 Essential matrix
    """
    E = K2.T @ F @ K1
    
    # Enforce the essential matrix constraints (singular values [1,1,0])
    U, S, Vt = np.linalg.svd(E)
    # Set the first two singular values to 1
    S = np.array([1.0, 1.0, 0.0])
    E = U @ np.diag(S) @ Vt
    
    logger.debug(
        f"Essential matrix: shape={E.shape}, "
        f"rank={np.linalg.matrix_rank(E)}, "
        f"singular values=[{S[0]:.4f}, {S[1]:.4f}, {S[2]:.4f}]"
    )
    
    return E


def decompose_E(E: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Decompose essential matrix into rotation and translation.

    Decomposes E into the four possible (R,t) configurations.

    Args:
        E: 3x3 Essential matrix

    Returns:
        List of four (R,t) tuples
    """
    # Compute the SVD of E
    U, S, Vt = np.linalg.svd(E)
    
    # Ensure proper rotation matrix with determinant 1
    if np.linalg.det(U) < 0:
        U = -U
    if np.linalg.det(Vt) < 0:
        Vt = -Vt
    
    # Define the W matrix
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    
    # Compute the possible rotations and translations
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t1 = U[:, 2].reshape(3, 1)  # last column of U
    t2 = -t1
    
    # Ensure rotation matrices (could have det = -1)
    if np.linalg.det(R1) < 0:
        R1 = -R1
        t1 = -t1
    if np.linalg.det(R2) < 0:
        R2 = -R2
        t2 = -t2
    
    # Return all four possible configurations
    poses = [(R1, t1), (R1, t2), (R2, t1), (R2, t2)]
    
    return poses


def triangulate_point(P1: np.ndarray, P2: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Triangulate a 3D point from two image points and camera matrices.

    Implements linear triangulation method using SVD.

    Args:
        P1: 3x4 projection matrix of the first camera
        P2: 3x4 projection matrix of the second camera
        x1: 2D point in the first image [x,y]
        x2: 2D point in the second image [x,y]

    Returns:
        Homogeneous coordinates of the 3D point [X,Y,Z,W]
    """
    # Create the design matrix A (4x4)
    A = np.zeros((4, 4))
    
    # Fill matrix A:
    # x1 * P1[2,:] - P1[0,:]
    # y1 * P1[2,:] - P1[1,:]
    # x2 * P2[2,:] - P2[0,:]
    # y2 * P2[2,:] - P2[1,:]
    A[0] = x1[0] * P1[2, :] - P1[0, :]
    A[1] = x1[1] * P1[2, :] - P1[1, :]
    A[2] = x2[0] * P2[2, :] - P2[0, :]
    A[3] = x2[1] * P2[2, :] - P2[1, :]
    
    # Compute SVD and extract the solution (last right singular vector)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    
    # Convert to homogeneous coordinates
    if X[3] != 0:
        X = X / X[3]
    
    return X


def check_cheirality(P1: np.ndarray, P2: np.ndarray, points_3d: np.ndarray) -> int:
    """Check cheirality constraint to determine correct camera pose.

    Counts how many triangulated points are in front of both cameras.

    Args:
        P1: 3x4 projection matrix of the first camera
        P2: 3x4 projection matrix of the second camera
        points_3d: Nx4 array of homogeneous 3D points

    Returns:
        Number of points that satisfy the cheirality constraint
    """
    # Extract rotation and translation from projection matrices
    R1 = P1[:3, :3]
    t1 = P1[:3, 3]
    R2 = P2[:3, :3]
    t2 = P2[:3, 3]
    
    # Count points with positive depth for both cameras
    n_valid = 0
    for X in points_3d:
        # Ensure homogeneous coordinates
        X_h = X
        if X[3] != 0:
            X_h = X / X[3]
        
        # Check if point is in front of first camera
        X_c1 = R1 @ X_h[:3] + t1
        z1 = X_c1[2]
        
        # Check if point is in front of second camera
        X_c2 = R2 @ X_h[:3] + t2
        z2 = X_c2[2]
        
        if z1 > 0 and z2 > 0:
            n_valid += 1
    
    return n_valid


def find_best_pose(poses: list[tuple[np.ndarray, np.ndarray]], 
                  K1: np.ndarray, K2: np.ndarray, 
                  matches: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find the best camera pose from the four possible E decompositions.

    Triangulates points with each pose and selects the one with most
    points in front of both cameras (satisfying cheirality constraint).

    Args:
        poses: List of four (R,t) tuples from E decomposition
        K1: Intrinsic matrix of the first camera
        K2: Intrinsic matrix of the second camera
        matches: Nx4 array of point correspondences [x1,y1,x2,y2]

    Returns:
        Tuple of (R, t, points_3d) where points_3d are the triangulated points
    """
    # First camera is at the origin with identity rotation
    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    
    # Initialize variables to track the best pose
    max_valid_points = -1
    best_P2 = None
    best_points_3d = None
    best_R = None
    best_t = None
    
    logger.debug(f"Testing {len(poses)} camera pose hypotheses")
    
    # For each possible pose
    for i, (R, t) in enumerate(poses):
        # Create projection matrix for the second camera
        P2 = K2 @ np.hstack((R, t))
        
        # Triangulate points for each match
        points_3d = []
        for j in range(matches.shape[0]):
            x1 = matches[j, :2]
            x2 = matches[j, 2:]
            X = triangulate_point(P1, P2, x1, x2)
            points_3d.append(X)
        
        points_3d = np.array(points_3d)
        
        # Check cheirality
        n_valid = check_cheirality(P1, P2, points_3d)
        logger.debug(f"Pose hypothesis {i+1}: {n_valid}/{len(points_3d)} valid points")
        
        if n_valid > max_valid_points:
            max_valid_points = n_valid
            best_P2 = P2
            best_points_3d = points_3d
            best_R = R
            best_t = t
    
    logger.info(
        f"Best pose has {max_valid_points}/{matches.shape[0]} "
        f"points satisfying cheirality ({max_valid_points/matches.shape[0]*100:.1f}%)"
    )
    
    return best_R, best_t, best_points_3d


def pnp_ransac(P3d: np.ndarray, p2d: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Solve the Perspective-n-Point problem using RANSAC.

    Implements EPnP with RANSAC and Gauss-Newton refinement.

    Args:
        P3d: Nx3 array of 3D points
        p2d: Nx2 array of corresponding 2D points
        K: 3x3 Intrinsic camera matrix

    Returns:
        3x4 Camera pose matrix [R|t]
    """
    start_time = time.perf_counter()
    
    # Convert to the format expected by OpenCV PnP functions
    object_points = P3d.reshape(-1, 1, 3).astype(np.float64)
    image_points = p2d.reshape(-1, 1, 2).astype(np.float64)
    
    # Use RANSAC to find camera pose robustly
    retval, rvec, tvec, inliers = cv2.solvePnPRansac(
        objectPoints=object_points,
        imagePoints=image_points,
        cameraMatrix=K,
        distCoeffs=None,
        flags=cv2.SOLVEPNP_EPNP,
        reprojectionError=2.0,
        iterationsCount=1000,
        confidence=0.99
    )
    
    if retval:
        # If successful, convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Create camera pose matrix [R|t]
        pose = np.hstack((R, tvec))
        
        n_inliers = len(inliers) if inliers is not None else 0
        inlier_ratio = n_inliers / len(P3d) if len(P3d) > 0 else 0
        
        # Refine using only inliers if we have enough
        if n_inliers >= 6:
            object_points_inliers = P3d[inliers].reshape(-1, 1, 3).astype(np.float64)
            image_points_inliers = p2d[inliers].reshape(-1, 1, 2).astype(np.float64)
            
            # Gauss-Newton refinement
            retval, rvec_refined, tvec_refined = cv2.solvePnP(
                objectPoints=object_points_inliers,
                imagePoints=image_points_inliers,
                cameraMatrix=K,
                distCoeffs=None,
                flags=cv2.SOLVEPNP_ITERATIVE,
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=True
            )
            
            if retval:
                R_refined, _ = cv2.Rodrigues(rvec_refined)
                pose = np.hstack((R_refined, tvec_refined))
                
                # Calculate reprojection error
                proj_points, _ = cv2.projectPoints(
                    object_points_inliers, rvec_refined, tvec_refined, K, None
                )
                proj_points = proj_points.reshape(-1, 2)
                errors = np.linalg.norm(proj_points - image_points_inliers.reshape(-1, 2), axis=1)
                rmse = np.sqrt(np.mean(errors**2))
                
                logger.debug(f"PnP refined RMSE: {rmse:.3f} pixels")
        
        elapsed_time = time.perf_counter() - start_time
        logger.info(
            f"PnP RANSAC: {n_inliers}/{len(P3d)} inliers ({inlier_ratio*100:.1f}%), "
            f"elapsed time: {elapsed_time:.3f}s"
        )
        
        return pose
    else:
        logger.warning("PnP RANSAC failed to find camera pose")
        return None
