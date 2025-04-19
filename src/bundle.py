"""Bundle adjustment module.

This module implements bundle adjustment optimization using the Schur
complement method to efficiently solve the sparse normal equations.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy import linalg, optimize, sparse
from tqdm import tqdm

from src.evaluate import reprojection_rmse

logger = logging.getLogger(__name__)


def rodrigues(r: np.ndarray) -> np.ndarray:
    """Convert rotation vector to rotation matrix.

    Args:
        r: 3x1 rotation vector

    Returns:
        3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    if theta < 1e-8:
        return np.eye(3)
    
    r = r / theta
    
    # Rodrigues formula
    K = np.array([
        [0, -r[2], r[1]],
        [r[2], 0, -r[0]],
        [-r[1], r[0], 0]
    ])
    
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
    return R


def rodrigues_jacobian(r: np.ndarray) -> np.ndarray:
    """Compute Jacobian of Rodrigues rotation.

    Args:
        r: 3x1 rotation vector

    Returns:
        9x3 Jacobian matrix (flattened dR/dr)
    """
    eps = 1e-6
    J = np.zeros((9, 3))
    
    # Compute Jacobian using finite differences
    for i in range(3):
        r_plus = r.copy()
        r_plus[i] += eps
        r_minus = r.copy()
        r_minus[i] -= eps
        
        R_plus = rodrigues(r_plus)
        R_minus = rodrigues(r_minus)
        
        # Derivative approximation
        dR = (R_plus - R_minus) / (2 * eps)
        J[:, i] = dR.flatten()
    
    return J


def project_point(point: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Project a 3D point onto an image plane.

    Args:
        point: 3D point coordinates
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        K: 3x3 camera intrinsic matrix

    Returns:
        2D projected point coordinates
    """
    # Convert 3D point to camera coordinates
    p_cam = R @ point + t.reshape(3)
    
    # Reject points behind the camera
    if p_cam[2] <= 0:
        return np.array([float('nan'), float('nan')])
    
    # Normalize by dividing by Z
    p_normalized = p_cam / p_cam[2]
    
    # Apply camera intrinsics
    p_img = K @ p_normalized
    
    return p_img[:2]


def compute_residuals(
    poses: np.ndarray,
    points: np.ndarray,
    observations: Dict[Tuple[int, int], np.ndarray],
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    K: np.ndarray
) -> np.ndarray:
    """Compute reprojection error residuals.

    Args:
        poses: Array of camera poses parameterized as [rx, ry, rz, tx, ty, tz]
        points: Array of 3D point coordinates
        observations: Dictionary mapping (camera_idx, point_idx) to 2D observations
        camera_indices: Array of camera indices for each observation
        point_indices: Array of point indices for each observation
        K: 3x3 camera intrinsic matrix

    Returns:
        Array of residuals (x_obs - x_proj, y_obs - y_proj) flattened
    """
    residuals = []
    
    for i in range(len(camera_indices)):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        
        # Get camera parameters
        r = poses[cam_idx, :3]
        t = poses[cam_idx, 3:].reshape(3, 1)
        
        # Convert from rotation vector to matrix
        R = rodrigues(r)
        
        # Get 3D point
        X = points[pt_idx]
        
        # Project point
        proj = project_point(X, R, t, K)
        
        # Get observed point
        obs = observations[(cam_idx, pt_idx)]
        
        # Compute residual
        residual = obs - proj
        residuals.append(residual)
    
    return np.array(residuals).flatten()


def compute_jacobian(
    poses: np.ndarray,
    points: np.ndarray,
    camera_indices: np.ndarray,
    point_indices: np.ndarray,
    K: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Jacobian matrices for bundle adjustment.

    Computes the Jacobian with respect to camera parameters (J_cam)
    and point coordinates (J_pts).

    Args:
        poses: Array of camera poses parameterized as [rx, ry, rz, tx, ty, tz]
        points: Array of 3D point coordinates
        camera_indices: Array of camera indices for each observation
        point_indices: Array of point indices for each observation
        K: 3x3 camera intrinsic matrix

    Returns:
        Tuple of (J_cam, J_pts) Jacobian matrices
    """
    n_observations = len(camera_indices)
    
    # Initialize Jacobians
    J_cam = np.zeros((2 * n_observations, 6 * poses.shape[0]))
    J_pts = np.zeros((2 * n_observations, 3 * points.shape[0]))
    
    for i in range(n_observations):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]
        
        # Get camera parameters
        r = poses[cam_idx, :3]
        t = poses[cam_idx, 3:].reshape(3, 1)
        
        # Convert from rotation vector to matrix
        R = rodrigues(r)
        
        # Get 3D point
        X = points[pt_idx]
        
        # Transform point to camera coordinates
        X_cam = R @ X + t.reshape(3)
        
        if X_cam[2] <= 0:
            continue  # Skip points behind the camera
        
        # Precompute values
        fx, fy = K[0, 0], K[1, 1]
        inv_z = 1.0 / X_cam[2]
        inv_z2 = inv_z * inv_z
        
        # Jacobian w.r.t. camera parameters (rotation)
        dR_dr = rodrigues_jacobian(r)
        
        # For each rotation parameter
        dr_block = np.zeros((2, 3))
        for j in range(3):
            dR_j = dR_dr[:, j].reshape(3, 3)
            dX_cam = dR_j @ X
            
            # Derivative of projection w.r.t. X_cam
            dr_block[0, j] = fx * (dX_cam[0] * inv_z - X_cam[0] * dX_cam[2] * inv_z2)
            dr_block[1, j] = fy * (dX_cam[1] * inv_z - X_cam[1] * dX_cam[2] * inv_z2)
        
        # Jacobian w.r.t. camera parameters (translation)
        dt_block = np.zeros((2, 3))
        dt_block[0, 0] = fx * inv_z
        dt_block[0, 2] = -fx * X_cam[0] * inv_z2
        dt_block[1, 1] = fy * inv_z
        dt_block[1, 2] = -fy * X_cam[1] * inv_z2
        
        # Fill in the camera Jacobian block
        J_cam[2*i:2*i+2, 6*cam_idx:6*cam_idx+3] = dr_block
        J_cam[2*i:2*i+2, 6*cam_idx+3:6*cam_idx+6] = dt_block
        
        # Jacobian w.r.t. point coordinates
        dX_block = np.zeros((2, 3))
        
        # Apply chain rule: d(proj)/dX = d(proj)/dX_cam * dX_cam/dX
        dX_block[0, 0] = fx * R[0, 0] * inv_z - fx * X_cam[0] * R[2, 0] * inv_z2
        dX_block[0, 1] = fx * R[0, 1] * inv_z - fx * X_cam[0] * R[2, 1] * inv_z2
        dX_block[0, 2] = fx * R[0, 2] * inv_z - fx * X_cam[0] * R[2, 2] * inv_z2
        
        dX_block[1, 0] = fy * R[1, 0] * inv_z - fy * X_cam[1] * R[2, 0] * inv_z2
        dX_block[1, 1] = fy * R[1, 1] * inv_z - fy * X_cam[1] * R[2, 1] * inv_z2
        dX_block[1, 2] = fy * R[1, 2] * inv_z - fy * X_cam[1] * R[2, 2] * inv_z2
        
        # Fill in the point Jacobian block
        J_pts[2*i:2*i+2, 3*pt_idx:3*pt_idx+3] = dX_block
    
    return J_cam, J_pts


def solve_normal_eq_schur(J_cam: np.ndarray, J_pts: np.ndarray, residuals: np.ndarray,
                          n_cameras: int, n_points: int, damping: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """Solve the normal equations using Schur complement.

    Efficiently solves the sparse bundle adjustment problem by eliminating
    the point parameters and solving for camera parameters first.

    Args:
        J_cam: Jacobian with respect to camera parameters
        J_pts: Jacobian with respect to point parameters
        residuals: Reprojection error residuals
        n_cameras: Number of cameras
        n_points: Number of 3D points
        damping: Damping factor for Levenberg-Marquardt

    Returns:
        Tuple of (delta_cameras, delta_points) parameter updates
    """
    # Compute the blocks of the normal equations
    JTJ_cc = J_cam.T @ J_cam
    JTJ_cp = J_cam.T @ J_pts
    JTJ_pp = J_pts.T @ J_pts
    
    # Add damping to the diagonal (Levenberg-Marquardt)
    for i in range(JTJ_cc.shape[0]):
        JTJ_cc[i, i] *= (1.0 + damping)
    for i in range(JTJ_pp.shape[0]):
        JTJ_pp[i, i] *= (1.0 + damping)
    
    # Compute right-hand side
    JTr_c = J_cam.T @ residuals
    JTr_p = J_pts.T @ residuals
    
    # Compute the Schur complement: S = JTJ_cc - JTJ_cp * inv(JTJ_pp) * JTJ_cp.T
    # First, compute block-diagonal inverse of JTJ_pp
    JTJ_pp_inv = np.zeros_like(JTJ_pp)
    for i in range(n_points):
        block = JTJ_pp[3*i:3*i+3, 3*i:3*i+3]
        JTJ_pp_inv[3*i:3*i+3, 3*i:3*i+3] = np.linalg.inv(block)
    
    # Compute Schur complement
    S = JTJ_cc - JTJ_cp @ JTJ_pp_inv @ JTJ_cp.T
    
    # Compute RHS of the reduced system
    reduced_rhs = JTr_c - JTJ_cp @ JTJ_pp_inv @ JTr_p
    
    # Solve for camera parameter update
    delta_cameras = np.linalg.solve(S, reduced_rhs)
    
    # Back-substitute to get point parameter update
    delta_points = np.zeros(3 * n_points)
    for i in range(n_points):
        block_rhs = JTr_p[3*i:3*i+3]
        
        # Extract relevant blocks from JTJ_cp.T for this point
        point_coupling = np.zeros(6 * n_cameras)
        for j in range(n_cameras):
            point_coupling[6*j:6*j+6] = JTJ_cp.T[3*i:3*i+3, 6*j:6*j+6].flatten()
        
        # Update block RHS with camera update
        block_rhs -= point_coupling @ delta_cameras
        
        # Solve for point update
        block = JTJ_pp[3*i:3*i+3, 3*i:3*i+3]
        delta_points[3*i:3*i+3] = np.linalg.solve(block, block_rhs)
    
    return delta_cameras, delta_points


def bundle_adjust(
    poses: List[np.ndarray],
    points3d: np.ndarray,
    observations: Dict[Tuple[int, int], np.ndarray],
    K: np.ndarray,
    fix_intrinsics: bool = True,
    max_iterations: int = 100,
    ftol: float = 1e-6,
    verbose: bool = True
) -> Tuple[List[np.ndarray], np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """Bundle adjustment optimization using Levenberg-Marquardt.

    Jointly optimizes camera poses and 3D points to minimize reprojection error.

    Args:
        poses: List of 3x4 camera pose matrices [R|t]
        points3d: Nx3 array of 3D points
        observations: Dictionary mapping (camera_idx, point_idx) to 2D observations
        K: Camera intrinsic matrix
        fix_intrinsics: Whether to keep intrinsics fixed
        max_iterations: Maximum number of optimization iterations
        ftol: Convergence threshold for relative change in error
        verbose: Whether to print progress information

    Returns:
        Tuple of (optimized_poses, optimized_points3d, observations)
    """
    start_time = time.perf_counter()
    logger.info(f"Starting bundle adjustment: {len(poses)} cameras, {len(points3d)} points")
    
    # Convert poses to parameter vector format [rx, ry, rz, tx, ty, tz]
    pose_params = np.zeros((len(poses), 6))
    for i, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Convert rotation matrix to rotation vector
        r, _ = cv2.Rodrigues(R)
        
        pose_params[i, :3] = r.flatten()
        pose_params[i, 3:] = t.flatten()
    
    # Create arrays for camera and point indices
    obs_list = []
    camera_indices = []
    point_indices = []
    
    for (cam_idx, pt_idx), obs in observations.items():
        obs_list.append(obs)
        camera_indices.append(cam_idx)
        point_indices.append(pt_idx)
    
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    obs_array = np.array(obs_list)
    
    # Define parameter vector and arrays for optimization
    n_cameras = len(poses)
    n_points = len(points3d)
    
    # Initial error
    initial_residuals = compute_residuals(
        pose_params, points3d, observations, camera_indices, point_indices, K
    )
    initial_error = np.sum(initial_residuals**2)
    initial_rmse = np.sqrt(initial_error / len(initial_residuals))
    
    logger.info(f"Initial reprojection RMSE: {initial_rmse:.4f} pixels")
    
    # Optimization parameters
    lambda_factor = 10.0  # Damping factor scaling
    lambda_ = 1e-3  # Initial damping value
    
    # Optimization loop
    error = initial_error
    converged = False
    
    range_obj = tqdm(range(max_iterations)) if verbose else range(max_iterations)
    for iteration in range_obj:
        # Compute residuals and Jacobian
        residuals = compute_residuals(
            pose_params, points3d, observations, camera_indices, point_indices, K
        )
        current_error = np.sum(residuals**2)
        
        # Check for convergence
        if abs(error - current_error) / error < ftol:
            converged = True
            logger.info(
                f"Bundle adjustment converged after {iteration+1} iterations "
                f"(relative change < {ftol})"
            )
            break
        
        # Update error for next iteration
        if current_error < error:
            # Decrease lambda (make more like Gauss-Newton)
            lambda_ /= lambda_factor
            error = current_error
        else:
            # Increase lambda (make more like gradient descent)
            lambda_ *= lambda_factor
            
            # Skip parameter update this iteration
            if verbose:
                tqdm.write(f"Iteration {iteration+1}: Error increased, trying again with lambda={lambda_:.2e}")
            continue
        
        # Compute Jacobian matrices
        J_cam, J_pts = compute_jacobian(
            pose_params, points3d, camera_indices, point_indices, K
        )
        
        # Solve normal equations using Schur complement
        delta_cameras, delta_points = solve_normal_eq_schur(
            J_cam, J_pts, residuals, n_cameras, n_points, damping=lambda_
        )
        
        # Update parameters
        pose_params = pose_params.reshape(-1) + delta_cameras.reshape(-1)
        pose_params = pose_params.reshape(n_cameras, 6)
        
        points3d = points3d.reshape(-1) + delta_points.reshape(-1)
        points3d = points3d.reshape(n_points, 3)
        
        # Report progress
        if verbose and (iteration % 5 == 0 or iteration == max_iterations - 1):
            current_rmse = np.sqrt(current_error / len(residuals))
            tqdm.write(
                f"Iteration {iteration+1}: RMSE = {current_rmse:.4f} px, "
                f"lambda = {lambda_:.2e}"
            )
    
    if not converged and verbose:
        logger.info(f"Bundle adjustment did not converge after {max_iterations} iterations")
    
    # Convert pose parameters back to 3x4 matrices
    optimized_poses = []
    for i in range(n_cameras):
        r = pose_params[i, :3]
        t = pose_params[i, 3:]
        
        R = rodrigues(r)
        pose = np.hstack((R, t.reshape(3, 1)))
        optimized_poses.append(pose)
    
    # Compute final RMSE
    final_residuals = compute_residuals(
        pose_params, points3d, observations, camera_indices, point_indices, K
    )
    final_error = np.sum(final_residuals**2)
    final_rmse = np.sqrt(final_error / len(final_residuals))
    
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        f"Bundle adjustment complete: {initial_rmse:.4f}px -> {final_rmse:.4f}px "
        f"({elapsed_time:.2f}s)"
    )
    
    return optimized_poses, points3d, observations


def optimize_poses_points(
    poses: List[np.ndarray],
    points3d: np.ndarray,
    observations: Dict[Tuple[int, int], np.ndarray],
    K: np.ndarray,
    fix_cameras_indices: Optional[List[int]] = None
) -> Tuple[List[np.ndarray], np.ndarray]:
    """Simplified wrapper around scipy.optimize for bundle adjustment.

    This is an alternative implementation using scipy's least_squares solver.

    Args:
        poses: List of 3x4 camera pose matrices [R|t]
        points3d: Nx3 array of 3D points
        observations: Dictionary mapping (camera_idx, point_idx) to 2D observations
        K: Camera intrinsic matrix
        fix_cameras_indices: Indices of cameras to keep fixed

    Returns:
        Tuple of (optimized_poses, optimized_points3d)
    """
    # Import here to make this function optional
    try:
        import cv2
    except ImportError:
        logger.error("OpenCV is required for optimize_poses_points")
        return poses, points3d
    
    # Convert data to parameter vector and observation vectors
    n_cameras = len(poses)
    n_points = len(points3d)
    
    # Default no fixed cameras
    if fix_cameras_indices is None:
        fix_cameras_indices = []
    
    # Camera parameters: (R, t) for each camera, in Rodrigues format for rotation
    camera_params = np.zeros((n_cameras, 6))
    for i, pose in enumerate(poses):
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Convert rotation matrix to rotation vector
        r, _ = cv2.Rodrigues(R)
        
        camera_params[i, :3] = r.flatten()
        camera_params[i, 3:] = t
    
    # Create arrays for camera and point indices
    cam_indices = []
    pt_indices = []
    observations_list = []
    
    for (cam_idx, pt_idx), obs in observations.items():
        cam_indices.append(cam_idx)
        pt_indices.append(pt_idx)
        observations_list.append(obs)
    
    cam_indices = np.array(cam_indices)
    pt_indices = np.array(pt_indices)
    observations_array = np.array(observations_list).flatten()
    
    # Function to compute residuals for scipy optimizer
    def bundle_residuals(params):
        """Compute residuals for the bundle adjustment problem."""
        # Extract camera and point parameters
        camera_params_local = params[:n_cameras * 6].reshape((n_cameras, 6))
        point_params_local = params[n_cameras * 6:].reshape((n_points, 3))
        
        # Replace fixed camera parameters with original values
        for idx in fix_cameras_indices:
            camera_params_local[idx] = camera_params[idx]
        
        # Compute projections and residuals
        residuals = np.zeros(2 * len(cam_indices))
        
        for i, (cam_idx, pt_idx) in enumerate(zip(cam_indices, pt_indices)):
            # Get camera and point
            cam = camera_params_local[cam_idx]
            point = point_params_local[pt_idx]
            
            # Convert camera parameters to rotation matrix and translation vector
            R = rodrigues(cam[:3])
            t = cam[3:].reshape(3, 1)
            
            # Project point
            proj = project_point(point, R, t, K)
            
            # Get observation
            obs = observations[(cam_idx, pt_idx)]
            
            # Compute residual
            residuals[2*i:2*i+2] = obs - proj
        
        return residuals
    
    # Initial parameter vector
    x0 = np.hstack((camera_params.flatten(), points3d.flatten()))
    
    # Run optimization
    logger.info(f"Running scipy bundle adjustment with {len(observations)} observations")
    start_time = time.perf_counter()
    
    # Create sparse matrix pattern for the Jacobian
    n_obs = len(cam_indices)
    n_params = len(x0)
    
    # Run optimization
    result = optimize.least_squares(
        bundle_residuals,
        x0,
        method='trf',  # Trust Region Reflective
        ftol=1e-4,
        xtol=1e-4,
        gtol=1e-4,
        x_scale='jac',
        loss='linear',
        max_nfev=100,
        verbose=2
    )
    
    # Extract optimized parameters
    optimized_params = result.x
    optimized_camera_params = optimized_params[:n_cameras * 6].reshape((n_cameras, 6))
    optimized_points = optimized_params[n_cameras * 6:].reshape((n_points, 3))
    
    # Replace fixed camera parameters with original values
    for idx in fix_cameras_indices:
        optimized_camera_params[idx] = camera_params[idx]
    
    # Convert back to pose matrices
    optimized_poses = []
    for i in range(n_cameras):
        r = optimized_camera_params[i, :3]
        t = optimized_camera_params[i, 3:]
        
        R = rodrigues(r)
        pose = np.hstack((R, t.reshape(3, 1)))
        optimized_poses.append(pose)
    
    elapsed_time = time.perf_counter() - start_time
    logger.info(f"Scipy bundle adjustment completed in {elapsed_time:.2f}s")
    
    # Compute final RMSE
    final_residuals = bundle_residuals(optimized_params)
    final_rmse = np.sqrt(np.mean(final_residuals**2))
    logger.info(f"Final reprojection RMSE: {final_rmse:.4f} pixels")
    
    return optimized_poses, optimized_points
