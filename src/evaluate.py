"""Evaluation metrics for 3D reconstruction.

This module implements evaluation metrics for 3D reconstruction quality,
including reprojection error, Chamfer distance, and timing utilities.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple, Union, Callable

import numpy as np
from scipy import spatial

logger = logging.getLogger(__name__)


def reprojection_rmse(
    poses: List[np.ndarray],
    points3d: np.ndarray,
    observations: Dict[Tuple[int, int], np.ndarray],
    K: np.ndarray
) -> float:
    """Calculate the root mean square reprojection error.

    Args:
        poses: List of camera poses [R|t]
        points3d: Nx3 array of 3D points
        observations: Dictionary mapping (camera_idx, point_idx) to 2D observations
        K: Camera intrinsic matrix

    Returns:
        Root mean square reprojection error in pixels
    """
    if not observations:
        logger.warning("No observations provided for reprojection error calculation")
        return float('inf')
    
    # Calculate reprojection errors for all observations
    squared_errors = []
    
    for (cam_idx, pt_idx), obs_pt in observations.items():
        # Skip if camera or point index is out of bounds
        if cam_idx >= len(poses) or pt_idx >= len(points3d):
            continue
        
        # Get camera pose
        pose = poses[cam_idx]
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Get 3D point
        point3d = points3d[pt_idx]
        
        # Project 3D point to camera
        point_cam = R @ point3d + t
        
        # Skip points behind the camera
        if point_cam[2] <= 0:
            continue
        
        # Project to image plane
        point_img = K @ (point_cam / point_cam[2])
        proj_pt = point_img[:2]
        
        # Calculate squared error
        error = obs_pt - proj_pt
        squared_error = np.sum(error ** 2)
        squared_errors.append(squared_error)
    
    if not squared_errors:
        logger.warning("No valid reprojections for error calculation")
        return float('inf')
    
    # Calculate RMSE
    mse = np.mean(squared_errors)
    rmse = np.sqrt(mse)
    
    return rmse


def chamfer_distance(pcd_est: np.ndarray, pcd_gt: np.ndarray) -> float:
    """Calculate the Chamfer distance between two point clouds.

    Args:
        pcd_est: Estimated point cloud, Nx3 array
        pcd_gt: Ground truth point cloud, Nx3 array

    Returns:
        Chamfer distance
    """
    if pcd_est.shape[0] == 0 or pcd_gt.shape[0] == 0:
        logger.warning("Empty point cloud provided for Chamfer distance calculation")
        return float('inf')
    
    # Create KDTrees for efficient nearest neighbor search
    tree_est = spatial.KDTree(pcd_est)
    tree_gt = spatial.KDTree(pcd_gt)
    
    # Calculate distances from estimated to ground truth
    distances_est_to_gt, _ = tree_gt.query(pcd_est, k=1)
    
    # Calculate distances from ground truth to estimated
    distances_gt_to_est, _ = tree_est.query(pcd_gt, k=1)
    
    # Calculate mean distances
    mean_est_to_gt = np.mean(distances_est_to_gt)
    mean_gt_to_est = np.mean(distances_gt_to_est)
    
    # Chamfer distance is the sum of the means
    chamfer_dist = mean_est_to_gt + mean_gt_to_est
    
    return chamfer_dist


def f1_score(pcd_est: np.ndarray, pcd_gt: np.ndarray, threshold: float = 0.1) -> float:
    """Calculate the F1 score for point cloud accuracy.

    Args:
        pcd_est: Estimated point cloud, Nx3 array
        pcd_gt: Ground truth point cloud, Nx3 array
        threshold: Distance threshold for considering a point a match

    Returns:
        F1 score (0 to 1, higher is better)
    """
    if pcd_est.shape[0] == 0 or pcd_gt.shape[0] == 0:
        logger.warning("Empty point cloud provided for F1 score calculation")
        return 0.0
    
    # Create KDTrees for efficient nearest neighbor search
    tree_est = spatial.KDTree(pcd_est)
    tree_gt = spatial.KDTree(pcd_gt)
    
    # Calculate distances
    distances_est_to_gt, _ = tree_gt.query(pcd_est, k=1)
    distances_gt_to_est, _ = tree_est.query(pcd_gt, k=1)
    
    # Calculate precision: TP / (TP + FP)
    true_positives_precision = np.sum(distances_est_to_gt < threshold)
    precision = true_positives_precision / len(pcd_est) if len(pcd_est) > 0 else 0
    
    # Calculate recall: TP / (TP + FN)
    true_positives_recall = np.sum(distances_gt_to_est < threshold)
    recall = true_positives_recall / len(pcd_gt) if len(pcd_gt) > 0 else 0
    
    # Calculate F1 score
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return f1


class Timer:
    """Utility class for timing operations with context manager support."""
    
    def __init__(self, name: str = "Timer", logger: Optional[logging.Logger] = None):
        """Initialize timer.
        
        Args:
            name: Timer name for logging
            logger: Logger to use (if None, uses module logger)
        """
        self.name = name
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
        self._timings = {}
    
    def start(self) -> None:
        """Start the timer."""
        self.start_time = time.perf_counter()
    
    def stop(self) -> float:
        """Stop the timer and return elapsed time.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            self.logger.warning(f"{self.name}: Timer stopped without being started")
            return 0.0
        
        self.end_time = time.perf_counter()
        elapsed = self.end_time - self.start_time
        self.logger.debug(f"{self.name}: {elapsed:.4f}s")
        return elapsed
    
    def __enter__(self) -> "Timer":
        """Start timing when entering context."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Stop timing when exiting context."""
        self.stop()
    
    def timeit(self, func: Callable) -> Callable:
        """Decorator to time a function.
        
        Args:
            func: Function to time
        
        Returns:
            Wrapped function
        """
        def wrapper(*args, **kwargs):
            with self:
                result = func(*args, **kwargs)
            return result
        
        return wrapper
    
    def lap(self, name: str) -> float:
        """Record a lap time with a given name.
        
        Args:
            name: Lap name
        
        Returns:
            Lap time in seconds
        """
        current_time = time.perf_counter()
        if self.start_time is None:
            self.start_time = current_time
        
        last_time = self._timings.get("__last", self.start_time)
        lap_time = current_time - last_time
        
        self._timings["__last"] = current_time
        self._timings[name] = lap_time
        
        self.logger.debug(f"{self.name} - {name}: {lap_time:.4f}s")
        return lap_time
    
    @property
    def timings(self) -> Dict[str, float]:
        """Get all recorded lap times.
        
        Returns:
            Dictionary of lap times
        """
        return {k: v for k, v in self._timings.items() if k != "__last"}
    
    @property
    def elapsed(self) -> float:
        """Get current elapsed time without stopping the timer.
        
        Returns:
            Elapsed time in seconds
        """
        if self.start_time is None:
            return 0.0
        
        return time.perf_counter() - self.start_time


class ReconstructionMetrics:
    """Class for calculating and storing reconstruction metrics."""
    
    def __init__(self):
        """Initialize metrics container."""
        self.metrics = {
            "n_images": 0,
            "n_sparse_points": 0,
            "rmse_reproj_px": None,
            "dense_points": 0,
            "runtime_s": 0.0,
            "stage_timings": {},
        }
    
    def update(self, metric_name: str, value: Union[int, float, Dict]) -> None:
        """Update a specific metric.
        
        Args:
            metric_name: Name of the metric to update
            value: New value for the metric
        """
        self.metrics[metric_name] = value
    
    def update_stage_timing(self, stage_name: str, time_s: float) -> None:
        """Update timing for a specific pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
            time_s: Time in seconds
        """
        self.metrics["stage_timings"][stage_name] = time_s
    
    def compute_sparse_metrics(
        self,
        poses: List[np.ndarray],
        points3d: np.ndarray,
        observations: Dict[Tuple[int, int], np.ndarray],
        K: np.ndarray
    ) -> None:
        """Compute metrics for sparse reconstruction.
        
        Args:
            poses: List of camera poses [R|t]
            points3d: Nx3 array of 3D points
            observations: Dictionary mapping (camera_idx, point_idx) to 2D observations
            K: Camera intrinsic matrix
        """
        self.metrics["n_images"] = len(poses)
        self.metrics["n_sparse_points"] = len(points3d)
        
        # Calculate reprojection error
        rmse = reprojection_rmse(poses, points3d, observations, K)
        self.metrics["rmse_reproj_px"] = rmse
    
    def compute_dense_metrics(self, pcd: np.ndarray, pcd_gt: Optional[np.ndarray] = None) -> None:
        """Compute metrics for dense reconstruction.
        
        Args:
            pcd: Dense point cloud, Nx3 or Nx6 array
            pcd_gt: Optional ground truth point cloud for comparison
        """
        self.metrics["dense_points"] = len(pcd)
        
        if pcd_gt is not None:
            # Extract just XYZ coordinates if color is included
            pcd_xyz = pcd[:, :3] if pcd.shape[1] > 3 else pcd
            pcd_gt_xyz = pcd_gt[:, :3] if pcd_gt.shape[1] > 3 else pcd_gt
            
            # Calculate Chamfer distance
            chamfer = chamfer_distance(pcd_xyz, pcd_gt_xyz)
            self.metrics["chamfer_distance"] = chamfer
            
            # Calculate F1 score
            f1 = f1_score(pcd_xyz, pcd_gt_xyz)
            self.metrics["f1_score"] = f1
    
    def to_dict(self) -> Dict:
        """Convert metrics to dictionary.
        
        Returns:
            Dictionary of metrics
        """
        return self.metrics.copy()
    
    def summary(self) -> str:
        """Generate a human-readable summary of metrics.
        
        Returns:
            Summary string
        """
        lines = [
            "Reconstruction Metrics:",
            f"  Images: {self.metrics['n_images']}",
            f"  Sparse points: {self.metrics['n_sparse_points']}",
        ]
        
        if self.metrics["rmse_reproj_px"] is not None:
            lines.append(f"  Reprojection RMSE: {self.metrics['rmse_reproj_px']:.4f} px")
        
        lines.append(f"  Dense points: {self.metrics['dense_points']}")
        
        if "chamfer_distance" in self.metrics:
            lines.append(f"  Chamfer distance: {self.metrics['chamfer_distance']:.4f}")
        
        if "f1_score" in self.metrics:
            lines.append(f"  F1 score: {self.metrics['f1_score']:.4f}")
        
        lines.append(f"  Total runtime: {self.metrics['runtime_s']:.2f}s")
        
        if self.metrics["stage_timings"]:
            lines.append("  Stage timings:")
            for stage, time_s in self.metrics["stage_timings"].items():
                lines.append(f"    {stage}: {time_s:.2f}s")
        
        return "\n".join(lines)
