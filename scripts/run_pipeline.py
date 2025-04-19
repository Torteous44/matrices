#!/usr/bin/env python3
"""
3D Reconstruction Pipeline

This script runs the complete 3D reconstruction pipeline from a folder of images
to a textured 3D mesh, implementing the Structure from Motion (SfM) and
Multi-View Stereo (MVS) algorithms.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import yaml
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import feature, geometry, bundle, dense, mesh, evaluate, visualise


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pipeline")


def load_config(config_path: Optional[str] = None) -> Dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
    
    # Load configuration from file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


def read_images(image_dir: str) -> List[np.ndarray]:
    """Read images from directory.

    Args:
        image_dir: Path to directory containing images

    Returns:
        List of images as numpy arrays
    """
    logger.info(f"Reading images from {image_dir}")
    
    # List image files
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(list(Path(image_dir).glob(ext)))
        image_files.extend(list(Path(image_dir).glob(ext.upper())))
    
    image_files.sort()
    
    if not image_files:
        logger.error(f"No images found in {image_dir}")
        sys.exit(1)
    
    # Read images
    images = []
    for image_file in tqdm(image_files, desc="Reading images"):
        image = cv2.imread(str(image_file))
        if image is None:
            logger.warning(f"Failed to read {image_file}")
            continue
        images.append(image)
    
    logger.info(f"Read {len(images)} images")
    
    # Print image dimensions
    if images:
        logger.info(f"Image dimensions: {images[0].shape}")
    
    return images


def estimate_camera_matrix(
    image_shape: Tuple[int, int],
    focal_length: Optional[float] = None
) -> np.ndarray:
    """Estimate camera intrinsic matrix from image dimensions.

    Args:
        image_shape: Image dimensions (height, width)
        focal_length: Optional focal length in pixels
            If not provided, it's estimated as max(width, height)

    Returns:
        3x3 camera intrinsic matrix
    """
    height, width = image_shape[:2]
    
    # Estimate focal length if not provided
    if focal_length is None:
        focal_length = max(width, height)
    
    # Create camera matrix
    K = np.array([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ])
    
    return K


def build_view_graph(
    matches: Dict[Tuple[int, int], np.ndarray],
    min_matches: int = 30
) -> np.ndarray:
    """Build view graph from feature matches.

    Args:
        matches: Dictionary mapping image pairs (i,j) to point correspondences
        min_matches: Minimum number of matches to consider a connection

    Returns:
        Binary adjacency matrix
    """
    # Get number of images
    image_indices = set()
    for i, j in matches.keys():
        image_indices.add(i)
        image_indices.add(j)
    n_images = max(image_indices) + 1
    
    # Create adjacency matrix
    adjacency = np.zeros((n_images, n_images), dtype=bool)
    
    # Fill adjacency matrix
    for (i, j), match_points in matches.items():
        if len(match_points) >= min_matches:
            adjacency[i, j] = True
            adjacency[j, i] = True
    
    return adjacency


def select_initial_pair(
    matches: Dict[Tuple[int, int], np.ndarray],
    adjacency: np.ndarray,
    min_matches: int = 100
) -> Tuple[int, int]:
    """Select initial image pair for incremental SfM.

    Args:
        matches: Dictionary mapping image pairs (i,j) to point correspondences
        adjacency: Binary adjacency matrix
        min_matches: Minimum number of matches for selection

    Returns:
        Tuple of (first_index, second_index)
    """
    # Find pairs with sufficient matches
    valid_pairs = []
    for (i, j), match_points in matches.items():
        if len(match_points) >= min_matches and adjacency[i, j]:
            valid_pairs.append((i, j, len(match_points)))
    
    if not valid_pairs:
        logger.error("No valid initial pair found")
        logger.info("Falling back to first pair with matches")
        first_pair = list(matches.keys())[0]
        return first_pair
    
    # Sort by number of matches (descending)
    valid_pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Return pair with most matches
    return valid_pairs[0][0], valid_pairs[0][1]


def incremental_sfm(
    images: List[np.ndarray],
    matches: Dict[Tuple[int, int], np.ndarray],
    K: np.ndarray,
    config: Dict
) -> Tuple[List[np.ndarray], np.ndarray, Dict[Tuple[int, int], np.ndarray]]:
    """Run incremental Structure from Motion.

    Args:
        images: List of input images
        matches: Dictionary mapping image pairs (i,j) to point correspondences
        K: Camera intrinsic matrix
        config: Configuration dictionary

    Returns:
        Tuple of (camera_poses, points3d, observations)
    """
    logger.info("Starting incremental Structure from Motion")
    
    # Get parameters from config
    min_inliers_initialization = config["sfm"]["min_inliers_initialization"]
    min_inliers_registration = config["sfm"]["min_inliers_registration"]
    global_bundle_freq = config["sfm"]["global_bundle_freq"]
    global_bundle_max_rmse = config["sfm"]["global_bundle_max_rmse"]
    
    # Build view graph
    adjacency = build_view_graph(matches, min_matches=min_inliers_initialization)
    
    # Select initial pair
    if config["sfm"]["init_pair_method"] == "max_inliers":
        idx1, idx2 = select_initial_pair(matches, adjacency, min_matches=min_inliers_initialization)
    else:
        # Sequential initialization
        for (i, j) in matches.keys():
            if len(matches[(i, j)]) >= min_inliers_initialization:
                idx1, idx2 = i, j
                break
        else:
            logger.error("No valid initial pair found")
            sys.exit(1)
    
    logger.info(f"Selected initial pair: ({idx1}, {idx2})")
    
    # Get matches for initial pair
    initial_matches = matches[(idx1, idx2)]
    
    # Calculate fundamental matrix
    F = geometry.fundamental_8pt(initial_matches)
    
    # Calculate essential matrix
    E = geometry.essential_from_F(F, K, K)
    
    # Decompose essential matrix to get camera poses
    pose_candidates = geometry.decompose_E(E)
    
    # First camera is at origin with identity rotation
    poses = [np.hstack((np.eye(3), np.zeros((3, 1))))]
    
    # Find best second camera pose using cheirality check
    R, t, initial_points3d = geometry.find_best_pose(pose_candidates, K, K, initial_matches)
    poses.append(np.hstack((R, t)))
    
    # Store homogeneous points
    points3d = initial_points3d[:, :3]  # Convert to non-homogeneous
    
    # Initialize observations dictionary
    observations = {}
    for i, point3d_idx in enumerate(range(len(points3d))):
        observations[(0, point3d_idx)] = initial_matches[i, :2]
        observations[(1, point3d_idx)] = initial_matches[i, 2:4]
    
    # Set up structures to track reconstruction state
    registered_images = {idx1, idx2}
    point3d_to_observations = {i: [(0, i), (1, i)] for i in range(len(points3d))}
    
    # Create queue of images to register
    remaining_images = list(set(range(len(images))) - registered_images)
    
    # Main incremental SfM loop
    while remaining_images:
        # Find best next image to register
        best_image_idx = None
        best_num_matches = 0
        
        for idx in remaining_images:
            # Count matches to already registered images
            num_matches = sum(
                len(matches.get((idx, reg_idx), []))
                for reg_idx in registered_images
                if (idx, reg_idx) in matches or (reg_idx, idx) in matches
            )
            
            if num_matches > best_num_matches:
                best_image_idx = idx
                best_num_matches = num_matches
        
        if best_image_idx is None:
            logger.warning("No more images can be registered")
            break
        
        logger.info(
            f"Registering image {best_image_idx} "
            f"({best_num_matches} matches to registered images)"
        )
        
        # Collect 2D-3D correspondences for PnP
        points2d = []
        points3d_idx = []
        
        for reg_idx in registered_images:
            # Check if match exists
            match_key = (
                (best_image_idx, reg_idx)
                if (best_image_idx, reg_idx) in matches
                else (reg_idx, best_image_idx)
                if (reg_idx, best_image_idx) in matches
                else None
            )
            
            if match_key is None:
                continue
            
            # Get matches between images
            match_points = matches[match_key]
            
            # If match_key is (reg_idx, best_image_idx), swap points
            if match_key[0] == reg_idx:
                # Swap to make it (best_image_idx, reg_idx)
                match_points = np.hstack((match_points[:, 2:4], match_points[:, :2]))
            
            # For each match, check if reg_idx has a 3D point
            for i, point in enumerate(match_points):
                for point3d_idx, obs_indices in point3d_to_observations.items():
                    # Check if registered image has this point
                    for obs_cam_idx, obs_point_idx in obs_indices:
                        if obs_cam_idx == registered_images.index(reg_idx):
                            # Compare coordinates
                            obs_point = observations[(obs_cam_idx, obs_point_idx)]
                            if np.allclose(obs_point, point[2:4], atol=1):
                                # Found a match, add to PnP correspondences
                                points2d.append(point[:2])
                                points3d_idx.append(point3d_idx)
                                break
                    else:
                        continue
                    break
        
        # Check if we have enough correspondences
        if len(points2d) < min_inliers_registration:
            logger.warning(
                f"Insufficient correspondences for image {best_image_idx} "
                f"({len(points2d)} < {min_inliers_registration})"
            )
            remaining_images.remove(best_image_idx)
            continue
        
        # Run PnP to estimate camera pose
        points2d_array = np.array(points2d)
        points3d_pnp = np.array([points3d[idx] for idx in points3d_idx])
        
        pose = geometry.pnp_ransac(points3d_pnp, points2d_array, K)
        
        if pose is None:
            logger.warning(f"PnP failed for image {best_image_idx}")
            remaining_images.remove(best_image_idx)
            continue
        
        # Add pose to list
        poses.append(pose)
        registered_images.add(best_image_idx)
        remaining_images.remove(best_image_idx)
        
        # Map from old to new image indices
        old_to_new = {idx: i for i, idx in enumerate(sorted(registered_images))}
        
        # Triangulate new points
        for reg_idx in registered_images - {best_image_idx}:
            # Check if match exists
            match_key = (
                (best_image_idx, reg_idx)
                if (best_image_idx, reg_idx) in matches
                else (reg_idx, best_image_idx)
                if (reg_idx, best_image_idx) in matches
                else None
            )
            
            if match_key is None:
                continue
            
            # Get matches between images
            match_points = matches[match_key]
            
            # If match_key is (reg_idx, best_image_idx), swap points
            if match_key[0] == reg_idx:
                # Swap to make it (best_image_idx, reg_idx)
                match_points = np.hstack((match_points[:, 2:4], match_points[:, :2]))
            
            # Get camera poses
            pose1 = poses[old_to_new[best_image_idx]]
            pose2 = poses[old_to_new[reg_idx]]
            
            # Create projection matrices
            P1 = K @ pose1
            P2 = K @ pose2
            
            # For each match, triangulate a new point if not already triangulated
            for i, point in enumerate(match_points):
                # Check if this match corresponds to an existing 3D point
                existing_point = False
                for point3d_idx, obs_indices in point3d_to_observations.items():
                    for obs_cam_idx, obs_point_idx in obs_indices:
                        if obs_cam_idx == old_to_new[reg_idx]:
                            obs_point = observations[(obs_cam_idx, obs_point_idx)]
                            if np.allclose(obs_point, point[2:4], atol=1):
                                # Add observation for this 3D point
                                observations[(old_to_new[best_image_idx], obs_point_idx)] = point[:2]
                                point3d_to_observations[point3d_idx].append((old_to_new[best_image_idx], obs_point_idx))
                                existing_point = True
                                break
                    if existing_point:
                        break
                
                if not existing_point:
                    # Triangulate new point
                    X = geometry.triangulate_point(P1, P2, point[:2], point[2:4])
                    
                    # Check if point is in front of both cameras
                    if geometry.check_cheirality(P1, P2, X.reshape(1, 4)) > 0:
                        # Add point to reconstruction
                        new_point3d_idx = len(points3d)
                        points3d = np.vstack((points3d, X[:3] / X[3]))
                        
                        # Add observations
                        observations[(old_to_new[best_image_idx], new_point3d_idx)] = point[:2]
                        observations[(old_to_new[reg_idx], new_point3d_idx)] = point[2:4]
                        
                        # Add to point3d_to_observations
                        point3d_to_observations[new_point3d_idx] = [
                            (old_to_new[best_image_idx], new_point3d_idx),
                            (old_to_new[reg_idx], new_point3d_idx)
                        ]
        
        # Run bundle adjustment if needed
        if (
            len(registered_images) % global_bundle_freq == 0
            or len(remaining_images) == 0
        ):
            logger.info(f"Running global bundle adjustment with {len(registered_images)} cameras")
            
            # Calculate current reprojection error
            rmse_before = evaluate.reprojection_rmse(poses, points3d, observations, K)
            logger.info(f"RMSE before bundle adjustment: {rmse_before:.4f} pixels")
            
            # Run bundle adjustment
            poses, points3d, observations = bundle.bundle_adjust(
                poses, points3d, observations, K,
                fix_intrinsics=True,
                max_iterations=50,
                verbose=True
            )
            
            # Calculate reprojection error after bundle adjustment
            rmse_after = evaluate.reprojection_rmse(poses, points3d, observations, K)
            logger.info(f"RMSE after bundle adjustment: {rmse_after:.4f} pixels")
            
            # Filter outlier points
            if rmse_after > global_bundle_max_rmse:
                logger.info("Filtering outliers after bundle adjustment")
                
                # Calculate reprojection errors for all observations
                squared_errors = {}
                
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
                    squared_errors[(cam_idx, pt_idx)] = squared_error
                
                # Calculate threshold for outlier rejection
                threshold = np.percentile(list(squared_errors.values()), 75) * 5.0
                
                # Remove outlier observations
                filtered_observations = {}
                filter_count = 0
                
                for (cam_idx, pt_idx), obs_pt in observations.items():
                    if (cam_idx, pt_idx) in squared_errors and squared_errors[(cam_idx, pt_idx)] < threshold:
                        filtered_observations[(cam_idx, pt_idx)] = obs_pt
                    else:
                        filter_count += 1
                
                logger.info(f"Filtered {filter_count} outlier observations")
                
                # Update observations
                observations = filtered_observations
                
                # Run bundle adjustment again after filtering
                poses, points3d, observations = bundle.bundle_adjust(
                    poses, points3d, observations, K,
                    fix_intrinsics=True,
                    max_iterations=25,
                    verbose=True
                )
                
                # Calculate reprojection error after filtering
                rmse_final = evaluate.reprojection_rmse(poses, points3d, observations, K)
                logger.info(f"RMSE after filtering: {rmse_final:.4f} pixels")
    
    logger.info(
        f"Incremental SfM complete: {len(registered_images)}/{len(images)} images registered, "
        f"{len(points3d)} points, {len(observations)} observations"
    )
    
    return poses, points3d, observations


def create_point_cloud_with_colors(
    points3d: np.ndarray,
    observations: Dict[Tuple[int, int], np.ndarray],
    images: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray
) -> np.ndarray:
    """Create colored point cloud from 3D points and observations.

    Args:
        points3d: Nx3 array of 3D points
        observations: Dictionary mapping (camera_idx, point_idx) to 2D observations
        images: List of input images
        poses: List of camera poses
        K: Camera intrinsic matrix

    Returns:
        Nx6 array of colored 3D points (XYZ + RGB)
    """
    logger.info("Creating colored point cloud")
    
    # Initialize colors
    colors = np.zeros((len(points3d), 3))
    color_counts = np.zeros(len(points3d))
    
    # For each observation, add color to corresponding 3D point
    for (cam_idx, pt_idx), obs_pt in observations.items():
        # Skip if camera or point index is out of bounds
        if cam_idx >= len(poses) or pt_idx >= len(points3d):
            continue
        
        # Get image and camera pose
        image = images[cam_idx]
        
        # Get image coordinates
        x, y = int(round(obs_pt[0])), int(round(obs_pt[1]))
        
        # Check if coordinates are within image bounds
        if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
            # Get color from image
            color = image[y, x]
            
            # Add to colors
            colors[pt_idx] += color
            color_counts[pt_idx] += 1
    
    # Average colors
    valid_mask = color_counts > 0
    colors[valid_mask] = colors[valid_mask] / color_counts[valid_mask].reshape(-1, 1)
    
    # For points with no valid observations, set default color
    colors[~valid_mask] = [200, 200, 200]  # Light gray
    
    # Create colored point cloud
    colored_points = np.hstack((points3d, colors))
    
    logger.info(f"Created colored point cloud with {len(colored_points)} points")
    
    return colored_points


def save_results(
    output_dir: str,
    poses: List[np.ndarray],
    points3d: np.ndarray,
    observations: Dict[Tuple[int, int], np.ndarray],
    K: np.ndarray,
    dense_cloud: Optional[np.ndarray] = None,
    mesh_obj: Optional[object] = None,
    metrics: Optional[Dict] = None
) -> None:
    """Save reconstruction results to output directory.

    Args:
        output_dir: Path to output directory
        poses: List of camera poses
        points3d: Nx3 array of 3D points
        observations: Dictionary mapping (camera_idx, point_idx) to 2D observations
        K: Camera intrinsic matrix
        dense_cloud: Dense point cloud (optional)
        mesh_obj: Triangle mesh (optional)
        metrics: Reconstruction metrics (optional)
    """
    logger.info(f"Saving results to {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save camera poses
    camera_file = os.path.join(output_dir, "cameras.npy")
    with open(camera_file, "wb") as f:
        np.save(f, np.array(poses))
    
    # Save camera intrinsics
    K_file = os.path.join(output_dir, "intrinsics.npy")
    with open(K_file, "wb") as f:
        np.save(f, K)
    
    # Save sparse point cloud
    points_file = os.path.join(output_dir, "points3d.npy")
    with open(points_file, "wb") as f:
        np.save(f, points3d)
    
    # Save observations
    observations_file = os.path.join(output_dir, "observations.json")
    with open(observations_file, "w") as f:
        # Convert keys to strings
        obs_dict = {f"{k[0]}_{k[1]}": v.tolist() for k, v in observations.items()}
        json.dump(obs_dict, f)
    
    # Save dense point cloud if available
    if dense_cloud is not None:
        dense_file = os.path.join(output_dir, "dense.npy")
        with open(dense_file, "wb") as f:
            np.save(f, dense_cloud)
    
    # Save mesh if available
    if mesh_obj is not None:
        mesh_file = os.path.join(output_dir, "mesh.obj")
        mesh.save_mesh(mesh_obj, mesh_file, "obj")
    
    # Save metrics if available
    if metrics is not None:
        metrics_file = os.path.join(output_dir, "report.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
    
    logger.info("Results saved successfully")


def run_pipeline(
    image_dir: str,
    output_dir: str,
    feature_method: str = "sift",
    dense_method: str = "mvs",
    visualise_results: bool = False,
    config_path: Optional[str] = None
) -> Dict:
    """Run the complete 3D reconstruction pipeline.

    Args:
        image_dir: Path to directory containing images
        output_dir: Path to output directory
        feature_method: Feature detection method (sift, orb)
        dense_method: Dense reconstruction method (mvs, nerf)
        visualise_results: Whether to visualize results
        config_path: Path to configuration file

    Returns:
        Dictionary of reconstruction metrics
    """
    # Start timer
    pipeline_timer = evaluate.Timer("Pipeline")
    pipeline_timer.start()
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up file logging
    file_handler = logging.FileHandler(os.path.join(output_dir, "log.txt"))
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    # Load configuration
    config = load_config(config_path)
    
    # Update configuration with command-line arguments
    config["feature"]["method"] = feature_method
    config["dense"]["method"] = dense_method
    config["io"]["input_dir"] = image_dir
    config["io"]["output_dir"] = output_dir
    
    # Initialize metrics
    metrics = evaluate.ReconstructionMetrics()
    
    # === Stage 1: Read Images ===
    with evaluate.Timer("Read Images") as timer:
        images = read_images(image_dir)
        metrics.update_stage_timing("read_images", timer.elapsed)
    
    # Get image dimensions
    height, width = images[0].shape[:2]
    
    # Estimate camera intrinsics
    K = estimate_camera_matrix(images[0].shape[:2])
    logger.info(f"Estimated camera matrix:\n{K}")
    
    # === Stage 2: Detect Features and Match ===
    with evaluate.Timer("Feature Detection and Matching") as timer:
        matches = feature.detect_and_match(
            images, 
            method=config["feature"]["method"],
            ratio=config["feature"]["matcher"]["ratio_test"]
        )
        metrics.update_stage_timing("detect_and_match", timer.elapsed)
    
    # === Stage 3: Build View Graph ===
    with evaluate.Timer("Build View Graph") as timer:
        adjacency = build_view_graph(
            matches, 
            min_matches=config["sfm"]["min_inliers_initialization"]
        )
        metrics.update_stage_timing("build_view_graph", timer.elapsed)
    
    # Visualize view graph
    camera_graph_path = os.path.join(output_dir, "camera_graph.png")
    # (Placeholder for camera positions in view graph visualization)
    
    # === Stage 4: Incremental SfM ===
    with evaluate.Timer("Incremental SfM") as timer:
        poses, points3d, observations = incremental_sfm(
            images, matches, K, config
        )
        metrics.update_stage_timing("incremental_sfm", timer.elapsed)
    
    # Create colored point cloud
    colored_points = create_point_cloud_with_colors(
        points3d, observations, images, poses, K
    )
    
    # Compute sparse reconstruction metrics
    metrics.compute_sparse_metrics(poses, points3d, observations, K)
    
    # Visualize sparse reconstruction
    if visualise_results:
        sparse_vis_path = os.path.join(output_dir, "sparse.png")
        visualise.create_reconstruction_summary_image(
            colored_points, None, None, poses, K, sparse_vis_path
        )
    
    # === Stage 5: Dense Reconstruction (optional) ===
    dense_cloud = None
    if dense_method in ["mvs", "nerf"]:
        with evaluate.Timer("Dense Reconstruction") as timer:
            dense_cloud = dense.run_dense(
                images, poses, K, mode=dense_method, config=config["dense"]
            )
            metrics.update_stage_timing("dense_reconstruction", timer.elapsed)
        
        # Compute dense reconstruction metrics
        metrics.compute_dense_metrics(dense_cloud)
        
        # Visualize dense reconstruction
        if visualise_results:
            dense_vis_path = os.path.join(output_dir, "dense.png")
            visualise.create_reconstruction_summary_image(
                colored_points, dense_cloud, None, poses, K, dense_vis_path
            )
    
    # === Stage 6: Mesh Reconstruction (optional) ===
    mesh_obj = None
    if dense_cloud is not None:
        with evaluate.Timer("Mesh Reconstruction") as timer:
            mesh_obj = mesh.create_textured_mesh(
                dense_cloud, images, poses, K, config["mesh"]
            )
            metrics.update_stage_timing("mesh_reconstruction", timer.elapsed)
        
        # Visualize mesh
        if visualise_results:
            mesh_vis_path = os.path.join(output_dir, "mesh.png")
            visualise.create_reconstruction_summary_image(
                None, None, mesh_obj, poses, K, mesh_vis_path
            )
    
    # === Stage 7: Evaluate and Save Results ===
    with evaluate.Timer("Save Results") as timer:
        # Update metrics
        metrics.update("runtime_s", pipeline_timer.elapsed)
        
        # Save metrics
        metrics_dict = metrics.to_dict()
        metrics_dict["datetime"] = datetime.datetime.now().isoformat()
        
        # Save results
        save_results(
            output_dir, poses, points3d, observations, K,
            dense_cloud, mesh_obj, metrics_dict
        )
        
        metrics.update_stage_timing("save_results", timer.elapsed)
    
    # Print metrics summary
    logger.info("\n" + metrics.summary())
    
    # === Stage 8: Interactive Visualization (optional) ===
    if visualise_results:
        with evaluate.Timer("Visualization") as timer:
            visualise.show(
                poses, K, colored_points, dense_cloud, mesh_obj,
                save_path=os.path.join(output_dir, "reconstruction.png")
            )
            metrics.update_stage_timing("visualization", timer.elapsed)
    
    # Return metrics
    return metrics.to_dict()


def main():
    """Main function to parse arguments and run the pipeline."""
    parser = argparse.ArgumentParser(description="3D Reconstruction Pipeline")
    parser.add_argument(
        "--images", "-i", dest="image_dir", required=True,
        help="Path to directory containing images"
    )
    parser.add_argument(
        "--output", "-o", dest="output_dir", default="results/run1",
        help="Path to output directory"
    )
    parser.add_argument(
        "--feature", "-f", dest="feature_method", default="sift",
        choices=["sift", "orb"],
        help="Feature detection method"
    )
    parser.add_argument(
        "--dense", "-d", dest="dense_method", default="mvs",
        choices=["none", "mvs", "nerf"],
        help="Dense reconstruction method"
    )
    parser.add_argument(
        "--visualise", "-v", dest="visualise", action="store_true",
        help="Visualize results"
    )
    parser.add_argument(
        "--config", "-c", dest="config_path", default=None,
        help="Path to configuration file"
    )
    
    args = parser.parse_args()
    
    # Set visualisation flag to False if dense method is 'none'
    if args.dense_method == "none":
        args.visualise = False
    
    # Run pipeline
    try:
        run_pipeline(
            args.image_dir,
            args.output_dir,
            args.feature_method,
            args.dense_method,
            args.visualise,
            args.config_path
        )
    except Exception as e:
        logger.exception(f"Error running pipeline: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
