"""Dense reconstruction module.

This module implements dense 3D reconstruction using either Multi-View Stereo (MVS)
or Neural Radiance Fields (NeRF).
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


def run_dense(
    images: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray,
    mode: str = "mvs",
    config: Optional[Dict] = None
) -> np.ndarray:
    """Run dense reconstruction to generate a dense point cloud.

    Args:
        images: List of input images
        poses: List of camera poses [R|t]
        K: Camera intrinsic matrix
        mode: Reconstruction mode, either "mvs" (Multi-View Stereo) or "nerf" (Neural Radiance Fields)
        config: Optional configuration parameters

    Returns:
        Dense point cloud as Nx6 array (XYZ + RGB)
    """
    if config is None:
        config = {}
    
    start_time = time.perf_counter()
    logger.info(f"Starting dense reconstruction using {mode.upper()}")
    
    if mode.lower() == "mvs":
        point_cloud = run_mvs(images, poses, K, config)
    elif mode.lower() == "nerf":
        point_cloud = run_nerf(images, poses, K, config)
    else:
        raise ValueError(f"Unknown dense reconstruction mode: {mode}")
    
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        f"Dense reconstruction complete: {point_cloud.shape[0]} points "
        f"(elapsed time: {elapsed_time:.2f}s)"
    )
    
    return point_cloud


def run_mvs(
    images: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray,
    config: Dict
) -> np.ndarray:
    """Run Multi-View Stereo to generate a dense point cloud.

    Uses OpenCV's PatchMatch Stereo implementation to compute depth maps
    and fuse them into a point cloud.

    Args:
        images: List of input images
        poses: List of camera poses [R|t]
        K: Camera intrinsic matrix
        config: Configuration parameters for MVS

    Returns:
        Dense point cloud as Nx6 array (XYZ + RGB)
    """
    # Get MVS parameters from config
    num_views = config.get("num_views", 5)
    window_size = config.get("window_size", 11)
    min_disp = config.get("min_disp", 0)
    max_disp = config.get("max_disp", 128)
    
    # Create PatchMatch Stereo object
    stereo = cv2.stereo.createRightMatcher(
        cv2.stereo.StereoBinarySGBM_create(
            minDisparity=min_disp,
            numDisparities=max_disp - min_disp,
            blockSize=window_size
        )
    )
    
    # Initialize empty point cloud
    points = []
    colors = []
    
    # For each image as reference
    for ref_idx in tqdm(range(len(images)), desc="MVS depth maps"):
        ref_img = images[ref_idx]
        ref_pose = poses[ref_idx]
        
        # Convert to grayscale if color
        if len(ref_img.shape) == 3:
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
        else:
            ref_gray = ref_img
        
        # Find best neighbor views
        neighbor_indices = find_best_neighbor_views(poses, ref_idx, num_views)
        
        if not neighbor_indices:
            continue
        
        # For each neighbor, compute depth map
        for neighbor_idx in neighbor_indices:
            neighbor_img = images[neighbor_idx]
            neighbor_pose = poses[neighbor_idx]
            
            # Convert to grayscale if color
            if len(neighbor_img.shape) == 3:
                neighbor_gray = cv2.cvtColor(neighbor_img, cv2.COLOR_BGR2GRAY)
            else:
                neighbor_gray = neighbor_img
            
            # Compute disparity
            disparity = compute_disparity(ref_gray, neighbor_gray, stereo)
            
            # Convert disparity to depth
            depth = disparity_to_depth(disparity, K, ref_pose, neighbor_pose)
            
            # Project depth map to 3D points
            pts, cols = depth_to_points(depth, ref_img, K, ref_pose)
            
            # Add to point cloud
            points.append(pts)
            colors.append(cols)
    
    # Concatenate all points and colors
    if points:
        points = np.vstack(points)
        colors = np.vstack(colors)
        
        # Combine points and colors
        point_cloud = np.hstack((points, colors))
    else:
        # Empty point cloud
        point_cloud = np.zeros((0, 6))
    
    return point_cloud


def find_best_neighbor_views(
    poses: List[np.ndarray],
    ref_idx: int,
    num_views: int
) -> List[int]:
    """Find the best neighbor views for a reference view.

    Selects views based on baseline and view angle.

    Args:
        poses: List of camera poses [R|t]
        ref_idx: Index of reference view
        num_views: Number of neighbor views to select

    Returns:
        List of indices of selected neighbor views
    """
    ref_pose = poses[ref_idx]
    ref_center = -ref_pose[:3, :3].T @ ref_pose[:3, 3]
    ref_forward = ref_pose[:3, :3].T @ np.array([0, 0, 1])
    
    # Compute scores for all other views
    scores = []
    for i, pose in enumerate(poses):
        if i == ref_idx:
            continue
        
        # Camera center
        center = -pose[:3, :3].T @ pose[:3, 3]
        
        # Baseline
        baseline = np.linalg.norm(center - ref_center)
        
        # Camera forward direction
        forward = pose[:3, :3].T @ np.array([0, 0, 1])
        
        # Angle between viewing directions
        angle = np.arccos(np.clip(np.dot(forward, ref_forward), -1.0, 1.0))
        
        # Score is based on baseline and view angle
        # Prefer moderate baseline and similar viewing direction
        score = baseline * np.exp(-angle * 2)
        scores.append((i, score))
    
    # Sort views by score
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Select top views
    return [idx for idx, _ in scores[:num_views]]


def compute_disparity(
    img1: np.ndarray,
    img2: np.ndarray,
    stereo: cv2.stereo.StereoMatcher
) -> np.ndarray:
    """Compute disparity map between two images.

    Args:
        img1: First image (grayscale)
        img2: Second image (grayscale)
        stereo: OpenCV stereo matcher

    Returns:
        Disparity map
    """
    # Ensure images have the same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Match images
    disparity = stereo.compute(img1, img2).astype(np.float32) / 16.0
    
    # Filter invalid disparities
    disparity[disparity < 0] = 0
    
    return disparity


def disparity_to_depth(
    disparity: np.ndarray,
    K: np.ndarray,
    pose1: np.ndarray,
    pose2: np.ndarray
) -> np.ndarray:
    """Convert disparity map to depth map.

    Args:
        disparity: Disparity map
        K: Camera intrinsic matrix
        pose1: Camera pose of the first image [R|t]
        pose2: Camera pose of the second image [R|t]

    Returns:
        Depth map
    """
    # Extract rotation and translation
    R1 = pose1[:3, :3]
    t1 = pose1[:3, 3]
    R2 = pose2[:3, :3]
    t2 = pose2[:3, 3]
    
    # Compute baseline
    center1 = -R1.T @ t1
    center2 = -R2.T @ t2
    baseline = np.linalg.norm(center2 - center1)
    
    # Get focal length
    f = K[0, 0]
    
    # Compute depth from disparity
    depth = np.zeros_like(disparity)
    valid_mask = disparity > 0
    depth[valid_mask] = f * baseline / disparity[valid_mask]
    
    # Filter outliers
    depth[depth > 100] = 0  # Max depth threshold
    
    return depth


def depth_to_points(
    depth: np.ndarray,
    image: np.ndarray,
    K: np.ndarray,
    pose: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert depth map to 3D points with colors.

    Args:
        depth: Depth map
        image: Color image
        K: Camera intrinsic matrix
        pose: Camera pose [R|t]

    Returns:
        Tuple of (points, colors)
    """
    # Get image dimensions
    h, w = depth.shape
    
    # Create pixel coordinates
    y, x = np.mgrid[0:h, 0:w]
    pixels = np.vstack((x.flatten(), y.flatten(), np.ones(x.size)))
    
    # Convert to camera coordinates
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    x = (pixels[0] - cx) / fx
    y = (pixels[1] - cy) / fy
    z = np.ones_like(x)
    
    # Scale by depth
    points_cam = np.vstack((x, y, z)) * depth.flatten()
    
    # Convert to world coordinates
    R = pose[:3, :3]
    t = pose[:3, 3]
    points_world = R.T @ (points_cam - t.reshape(3, 1))
    
    # Get RGB colors
    if len(image.shape) == 3:
        colors = image.reshape(-1, 3)
    else:
        # Grayscale to RGB
        colors = np.repeat(image.reshape(-1, 1), 3, axis=1)
    
    # Filter out invalid points (zero depth)
    valid_mask = depth.flatten() > 0
    valid_points = points_world[:, valid_mask].T
    valid_colors = colors[valid_mask]
    
    return valid_points, valid_colors


def run_nerf(
    images: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray,
    config: Dict
) -> np.ndarray:
    """Run Neural Radiance Fields (NeRF) to generate a dense point cloud.

    Implements a minimal version of NeRF with positional encoding.

    Args:
        images: List of input images
        poses: List of camera poses [R|t]
        K: Camera intrinsic matrix
        config: Configuration parameters for NeRF

    Returns:
        Dense point cloud as Nx6 array (XYZ + RGB)
    """
    # Try to import necessary libraries
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError:
        logger.error("PyTorch is required for NeRF reconstruction")
        return np.zeros((0, 6))
    
    # Get NeRF parameters from config
    encoding_level = config.get("encoding_level", 6)
    n_layers = config.get("n_layers", 4)
    hidden_dim = config.get("hidden_dim", 128)
    n_iters = config.get("n_iters", 1000)
    batch_size = config.get("batch_size", 256)
    
    # Normalize images to [0, 1]
    norm_images = [img.astype(np.float32) / 255.0 for img in images]
    
    # Get image dimensions
    h, w = norm_images[0].shape[:2]
    
    # Define NeRF model
    class NeRF(nn.Module):
        def __init__(self, input_dim=3, encoding_level=6, hidden_dim=128, output_dim=4):
            super(NeRF, self).__init__()
            
            # Positional encoding dimensions
            self.encoding_level = encoding_level
            self.input_dim = input_dim
            encoded_dim = input_dim * (1 + 2 * encoding_level)
            
            # MLP layers
            self.layers = nn.ModuleList([
                nn.Linear(encoded_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            ])
        
        def positional_encoding(self, x):
            """Apply positional encoding to input."""
            encodings = [x]
            
            for i in range(self.encoding_level):
                freq = 2.0 ** i
                encodings.append(torch.sin(freq * x))
                encodings.append(torch.cos(freq * x))
            
            return torch.cat(encodings, dim=-1)
        
        def forward(self, x):
            """Forward pass through the network."""
            x = self.positional_encoding(x)
            
            for layer in self.layers:
                x = layer(x)
            
            # Split output into RGB and density
            rgb = torch.sigmoid(x[..., :3])  # [0, 1]
            density = F.relu(x[..., 3])  # Non-negative
            
            return rgb, density
    
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeRF(encoding_level=encoding_level, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # Prepare camera parameters
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Direction vectors for each pixel
    i, j = torch.meshgrid(
        torch.linspace(0, h-1, h),
        torch.linspace(0, w-1, w)
    )
    
    directions = torch.stack([
        (j - cx) / fx,
        -(i - cy) / fy,
        -torch.ones_like(i)
    ], dim=-1).to(device)
    
    # Convert poses to torch tensors
    torch_poses = []
    for pose in poses:
        torch_pose = torch.tensor(pose, dtype=torch.float32).to(device)
        torch_poses.append(torch_pose)
    
    # Convert images to torch tensors
    torch_images = []
    for img in norm_images:
        if len(img.shape) == 2:
            # Convert grayscale to RGB
            img = np.stack([img, img, img], axis=-1)
        torch_img = torch.tensor(img, dtype=torch.float32).to(device)
        torch_images.append(torch_img)
    
    # Training loop
    logger.info(f"Training NeRF: {n_iters} iterations")
    for iteration in tqdm(range(n_iters), desc="Training NeRF"):
        # Randomly select an image
        img_idx = np.random.randint(len(images))
        pose = torch_poses[img_idx]
        target_img = torch_images[img_idx]
        
        # Randomly select pixels
        select_inds = np.random.choice(h * w, size=batch_size, replace=False)
        select_i = select_inds // w
        select_j = select_inds % w
        
        # Get rays for selected pixels
        rays_o = pose[:3, 3].expand(batch_size, 3)  # Origin
        rays_d = directions[select_i, select_j].reshape(-1, 3)  # Direction
        rays_d = torch.sum(rays_d[..., None, :] * pose[:3, :3], dim=-1)
        
        # Normalize ray directions
        rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
        
        # Sample points along rays
        n_samples = 64
        near, far = 0.1, 10.0
        t_vals = torch.linspace(0., 1., n_samples).to(device)
        z_vals = near * (1. - t_vals) + far * t_vals
        
        # Add noise to sampling
        z_vals = z_vals.expand(batch_size, n_samples)
        noise = (torch.rand(batch_size, n_samples) - 0.5) * (far - near) / n_samples
        z_vals = z_vals + noise.to(device)
        
        # Calculate 3D sample points
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[:, :, None]
        
        # Flatten for the network
        pts_flat = pts.reshape(-1, 3)
        
        # Run network
        rgb_flat, density_flat = model(pts_flat)
        
        # Reshape
        rgb = rgb_flat.reshape(batch_size, n_samples, 3)
        density = density_flat.reshape(batch_size, n_samples)
        
        # Calculate weights for volumetric rendering
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat([dists, torch.ones_like(dists[:, :1]) * 1e-3], dim=-1)
        
        alpha = 1. - torch.exp(-density * dists)
        weights = alpha * torch.cumprod(
            torch.cat([torch.ones_like(alpha[:, :1]), 1. - alpha + 1e-10], dim=-1),
            dim=-1
        )[:, :-1]
        
        # Calculate rendered color
        rgb_rendered = torch.sum(weights[:, :, None] * rgb, dim=-2)
        
        # Target pixel colors
        target_rgb = target_img[select_i, select_j]
        
        # Calculate loss
        loss = F.mse_loss(rgb_rendered, target_rgb)
        
        # Backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (iteration % 100) == 0:
            logger.debug(f"Iteration {iteration}: loss = {loss.item():.6f}")
    
    # Extract dense point cloud
    logger.info("Extracting point cloud from NeRF")
    
    # Define volume bounds
    x_range = torch.linspace(-1, 1, 64).to(device)
    y_range = torch.linspace(-1, 1, 64).to(device)
    z_range = torch.linspace(-1, 1, 64).to(device)
    
    # Create grid
    grid_x, grid_y, grid_z = torch.meshgrid(x_range, y_range, z_range)
    grid_pts = torch.stack([grid_x, grid_y, grid_z], dim=-1).reshape(-1, 3)
    
    # Evaluate network in batches
    batch_size = 1024
    n_batches = (grid_pts.shape[0] + batch_size - 1) // batch_size
    
    all_rgb = []
    all_density = []
    
    with torch.no_grad():
        for i in range(n_batches):
            start = i * batch_size
            end = min((i + 1) * batch_size, grid_pts.shape[0])
            
            batch_pts = grid_pts[start:end]
            rgb, density = model(batch_pts)
            
            all_rgb.append(rgb)
            all_density.append(density)
    
    all_rgb = torch.cat(all_rgb, dim=0)
    all_density = torch.cat(all_density, dim=0)
    
    # Extract points with high density
    threshold = torch.quantile(all_density, 0.95)
    mask = all_density > threshold
    
    if torch.sum(mask) == 0:
        logger.warning("No points with high enough density found")
        return np.zeros((0, 6))
    
    points = grid_pts[mask].cpu().numpy()
    colors = all_rgb[mask].cpu().numpy() * 255
    
    # Combine points and colors
    point_cloud = np.hstack((points, colors))
    
    return point_cloud


# Placeholder for sample functions below (TODOs)

def patchmatch_stereo() -> None:
    """TODO: Implement OpenCV's patchMatchStereo.compute."""
    pass


def multiple_lens_intrinsics_optimization() -> None:
    """TODO: Implement optimization for multiple lens intrinsics."""
    pass


def streaming_slam_mode() -> None:
    """TODO: Implement streaming (online) SLAM mode."""
    pass
