#!/usr/bin/env python3
"""
Neural Radiance Field (NeRF) Training

This script implements a minimal NeRF model for novel view synthesis,
using positional encoding and a simple MLP architecture.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import evaluate, visualise


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("nerf")


class NeRF(nn.Module):
    """Neural Radiance Field model.
    
    A minimal implementation of NeRF using positional encoding
    and a simple MLP architecture.
    """
    
    def __init__(
        self,
        pos_encoding_levels: int = 10,
        dir_encoding_levels: int = 4,
        hidden_dim: int = 256,
        n_layers: int = 8,
        skips: List[int] = [4],
    ):
        """Initialize NeRF model.
        
        Args:
            pos_encoding_levels: Number of frequency levels for positional encoding
            dir_encoding_levels: Number of frequency levels for directional encoding
            hidden_dim: Dimension of hidden layers
            n_layers: Number of hidden layers
            skips: List of indices where to add skip connections
        """
        super().__init__()
        
        # Encoding dimensions
        self.pos_encoding_levels = pos_encoding_levels
        self.dir_encoding_levels = dir_encoding_levels
        
        # Input dimensions after encoding
        pos_encoded_dim = 3 + 3 * 2 * pos_encoding_levels  # Original + sin/cos for each freq
        dir_encoded_dim = 3 + 3 * 2 * dir_encoding_levels
        
        # Create MLP layers
        self.skips = skips
        self.pos_layers = nn.ModuleList()
        
        # First layer (position encoding -> hidden)
        self.pos_layers.append(nn.Linear(pos_encoded_dim, hidden_dim))
        
        # Hidden layers with skip connections
        for i in range(n_layers - 1):
            if i in skips:
                # Add skip connection from input
                self.pos_layers.append(nn.Linear(hidden_dim + pos_encoded_dim, hidden_dim))
            else:
                self.pos_layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        # Output layers
        self.density_layer = nn.Linear(hidden_dim, 1)  # Density (sigma)
        
        # Direction dependent layers for color prediction
        self.feat_layer = nn.Linear(hidden_dim, hidden_dim)
        self.dir_layer = nn.Linear(dir_encoded_dim + hidden_dim, hidden_dim // 2)
        self.color_layer = nn.Linear(hidden_dim // 2, 3)
    
    def positional_encoding(self, x: torch.Tensor, levels: int) -> torch.Tensor:
        """Apply positional encoding to input.
        
        Args:
            x: Input tensor of shape (..., 3)
            levels: Number of frequency levels
        
        Returns:
            Encoded tensor of shape (..., 3 + 3*2*levels)
        """
        # Original values
        encoded = [x]
        
        # Apply encoding: sin(2^i * pi * x) and cos(2^i * pi * x)
        for i in range(levels):
            freq = 2.0 ** i
            encoded.append(torch.sin(freq * x))
            encoded.append(torch.cos(freq * x))
        
        # Concatenate all encodings
        return torch.cat(encoded, dim=-1)
    
    def forward(
        self, positions: torch.Tensor, directions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            positions: 3D positions of shape (..., 3)
            directions: View directions of shape (..., 3)
        
        Returns:
            Tuple of (rgb, density) of shapes (..., 3) and (..., 1)
        """
        # Normalize directions
        directions = F.normalize(directions, dim=-1)
        
        # Apply positional encoding
        pos_encoded = self.positional_encoding(positions, self.pos_encoding_levels)
        dir_encoded = self.positional_encoding(directions, self.dir_encoding_levels)
        
        # Forward pass for density
        h = pos_encoded
        for i, layer in enumerate(self.pos_layers):
            if i in self.skips:
                h = torch.cat([h, pos_encoded], dim=-1)
            h = F.relu(layer(h))
        
        # Density output
        density = F.relu(self.density_layer(h))
        
        # Forward pass for color
        feature = self.feat_layer(h)
        h = torch.cat([feature, dir_encoded], dim=-1)
        h = F.relu(self.dir_layer(h))
        rgb = torch.sigmoid(self.color_layer(h))  # Sigmoid to ensure [0, 1]
        
        return rgb, density


class RayGenerator:
    """Generate rays for volume rendering.
    
    This class generates rays emanating from a camera and passing through pixels.
    """
    
    def __init__(self, H: int, W: int, K: np.ndarray):
        """Initialize ray generator.
        
        Args:
            H: Image height
            W: Image width
            K: Camera intrinsic matrix (3x3)
        """
        self.H = H
        self.W = W
        self.focal_x = K[0, 0]
        self.focal_y = K[1, 1]
        self.cx = K[0, 2]
        self.cy = K[1, 2]
    
    def get_rays(self, pose: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate rays for all pixels in an image.
        
        Args:
            pose: Camera pose matrix (3x4 or 4x4) [R|t]
        
        Returns:
            Tuple of (origins, directions) for rays, each of shape (H*W, 3)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Extract rotation and translation
        if pose.shape[0] == 3:
            R = pose[:3, :3]
            t = pose[:3, 3]
        else:
            R = pose[:3, :3]
            t = pose[:3, 3]
        
        # Create pixel coordinates
        i, j = torch.meshgrid(
            torch.arange(self.H, device=device),
            torch.arange(self.W, device=device),
            indexing="ij"
        )
        i = i.flatten()
        j = j.flatten()
        
        # Convert to normalized coordinates
        x = (j - self.cx) / self.focal_x
        y = (i - self.cy) / self.focal_y
        z = -torch.ones_like(x)  # Points along -Z axis in camera coordinates
        
        # Stack coordinates
        dirs = torch.stack([x, y, z], dim=-1)
        
        # Transform to world coordinates
        R_torch = torch.from_numpy(R).to(device)
        dirs_world = torch.matmul(dirs, R_torch.T)
        
        # Normalize directions
        dirs_world = F.normalize(dirs_world, dim=-1)
        
        # Create ray origins (camera position)
        origins = torch.from_numpy(t).to(device).expand(dirs_world.shape[0], 3)
        
        return origins, dirs_world
    
    def get_batch_rays(
        self, pose: np.ndarray, n_rays: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of randomly sampled rays.
        
        Args:
            pose: Camera pose matrix (3x4 or 4x4) [R|t]
            n_rays: Number of rays to sample
        
        Returns:
            Tuple of (origins, directions) for rays, each of shape (n_rays, 3)
        """
        # Get all rays
        origins, directions = self.get_rays(pose)
        
        # Randomly select n_rays
        indices = torch.randint(0, origins.shape[0], (n_rays,), device=origins.device)
        
        return origins[indices], directions[indices]


class NeRFDataset(Dataset):
    """Dataset for training NeRF.
    
    This dataset provides ray batches and corresponding RGB values.
    """
    
    def __init__(
        self,
        images: List[np.ndarray],
        poses: List[np.ndarray],
        K: np.ndarray,
        n_rays: int = 1024,
    ):
        """Initialize NeRF dataset.
        
        Args:
            images: List of RGB images
            poses: List of camera poses
            K: Camera intrinsic matrix (3x3)
            n_rays: Number of rays to sample per batch
        """
        super().__init__()
        self.images = images
        self.poses = poses
        self.K = K
        self.n_rays = n_rays
        
        # Get image dimensions
        self.H, self.W = images[0].shape[:2]
        
        # Create ray generator
        self.ray_generator = RayGenerator(self.H, self.W, self.K)
        
        # Convert images to torch tensors, normalized to [0, 1]
        self.images_torch = []
        for img in images:
            if img.dtype == np.uint8:
                img = img.astype(np.float32) / 255.0
            self.images_torch.append(torch.from_numpy(img))
        
        # Pre-generate ray indices for each image
        self.rays_per_img = n_rays // len(images)
        self.ray_indices = []
        for _ in range(len(images)):
            self.ray_indices.append(
                torch.randint(0, self.H * self.W, (self.rays_per_img,))
            )
    
    def __len__(self) -> int:
        """Return number of batches."""
        return 1000  # Arbitrary large number for training iterations
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a batch of rays and RGB values.
        
        Args:
            idx: Batch index (unused, random sampling)
        
        Returns:
            Tuple of (origins, directions, rgb_values)
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Collect rays and RGB values
        all_origins = []
        all_directions = []
        all_rgb = []
        
        # Sample rays from each image
        for img_idx in range(len(self.images)):
            # Get pre-generated ray indices
            ray_idx = self.ray_indices[img_idx]
            
            # Generate new random indices
            self.ray_indices[img_idx] = torch.randint(
                0, self.H * self.W, (self.rays_per_img,), device=device
            )
            
            # Get rays
            origins, directions = self.ray_generator.get_rays(self.poses[img_idx])
            
            # Get RGB values
            rgb = self.images_torch[img_idx].reshape(-1, 3).to(device)
            
            # Sample rays
            all_origins.append(origins[ray_idx])
            all_directions.append(directions[ray_idx])
            all_rgb.append(rgb[ray_idx])
        
        # Concatenate
        origins = torch.cat(all_origins, dim=0)
        directions = torch.cat(all_directions, dim=0)
        rgb = torch.cat(all_rgb, dim=0)
        
        return origins, directions, rgb


def render_rays(
    model: nn.Module,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
    perturb: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Render rays using volume rendering.
    
    Args:
        model: NeRF model
        ray_origins: Ray origins of shape (n_rays, 3)
        ray_directions: Ray directions of shape (n_rays, 3)
        near: Near plane distance
        far: Far plane distance
        n_samples: Number of samples per ray
        perturb: Whether to perturb samples
    
    Returns:
        Tuple of (rgb, depth) of shapes (n_rays, 3) and (n_rays,)
    """
    # Generate sample points along rays
    t_vals = torch.linspace(
        near, far, n_samples, device=ray_origins.device
    )
    
    # Reshape for batched computation
    t_vals = t_vals.expand(ray_origins.shape[0], n_samples)
    
    # Add noise to samples
    if perturb:
        mids = 0.5 * (t_vals[:, 1:] + t_vals[:, :-1])
        upper = torch.cat([mids, t_vals[:, -1:]], dim=-1)
        lower = torch.cat([t_vals[:, :1], mids], dim=-1)
        t_rand = torch.rand(t_vals.shape, device=t_vals.device)
        t_vals = lower + (upper - lower) * t_rand
    
    # Generate sample positions along rays
    # (n_rays, n_samples, 3)
    ray_directions_expanded = ray_directions.unsqueeze(1).expand(-1, n_samples, -1)
    ray_origins_expanded = ray_origins.unsqueeze(1).expand(-1, n_samples, -1)
    sample_points = ray_origins_expanded + t_vals.unsqueeze(-1) * ray_directions_expanded
    
    # Reshape for batched forward pass
    flattened_points = sample_points.reshape(-1, 3)
    flattened_dirs = ray_directions_expanded.reshape(-1, 3)
    
    # Run forward pass
    rgb, density = model(flattened_points, flattened_dirs)
    
    # Reshape outputs
    rgb = rgb.reshape(ray_origins.shape[0], n_samples, 3)
    density = density.reshape(ray_origins.shape[0], n_samples, 1)
    
    # Calculate distances between samples
    dists = t_vals[:, 1:] - t_vals[:, :-1]
    dists = torch.cat(
        [dists, torch.ones_like(dists[:, :1]) * 1e-3], dim=-1
    ).unsqueeze(-1)
    
    # Calculate alpha compositing weights
    alpha = 1.0 - torch.exp(-density * dists)
    
    # Calculate transmittance (probability of ray traveling without hitting anything)
    weights = alpha * torch.cumprod(
        torch.cat(
            [torch.ones_like(alpha[:, :1]), 1.0 - alpha + 1e-10], dim=-1
        ),
        dim=-2
    )[:, :-1]
    
    # Compute final RGB and depth
    rgb_final = torch.sum(weights * rgb, dim=1)
    depth_final = torch.sum(weights * t_vals.unsqueeze(-1), dim=1).squeeze(-1)
    
    return rgb_final, depth_final


def train_nerf(
    images: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray,
    config: Dict,
    output_dir: str,
) -> nn.Module:
    """Train a NeRF model.
    
    Args:
        images: List of RGB images
        poses: List of camera poses
        K: Camera intrinsic matrix (3x3)
        config: Configuration dictionary
        output_dir: Output directory
    
    Returns:
        Trained NeRF model
    """
    # Get parameters from config
    n_epochs = config.get("n_epochs", 100)
    batch_size = config.get("batch_size", 1024)
    learning_rate = config.get("learning_rate", 5e-4)
    pos_encoding_levels = config.get("encoding_level", 10)
    dir_encoding_levels = config.get("dir_encoding_level", 4)
    hidden_dim = config.get("hidden_dim", 256)
    n_layers = config.get("n_layers", 8)
    n_samples = config.get("n_samples", 64)
    near = config.get("near", 2.0)
    far = config.get("far", 6.0)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    file_handler = logging.FileHandler(os.path.join(output_dir, "nerf_training.log"))
    file_handler.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Create NeRF model
    model = NeRF(
        pos_encoding_levels=pos_encoding_levels,
        dir_encoding_levels=dir_encoding_levels,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create dataset and dataloader
    dataset = NeRFDataset(images, poses, K, n_rays=batch_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Training loop
    with evaluate.Timer("Training") as timer:
        for epoch in range(n_epochs):
            epoch_losses = []
            
            with tqdm(total=len(dataloader), desc=f"Epoch {epoch+1}/{n_epochs}") as pbar:
                for batch_idx, (origins, directions, target_rgb) in enumerate(dataloader):
                    # Forward pass
                    rgb, _ = render_rays(
                        model,
                        origins.squeeze(0),
                        directions.squeeze(0),
                        near=near,
                        far=far,
                        n_samples=n_samples,
                        perturb=True,
                    )
                    
                    # Calculate loss
                    loss = F.mse_loss(rgb, target_rgb.squeeze(0))
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update progress bar
                    epoch_losses.append(loss.item())
                    pbar.set_postfix(loss=f"{loss.item():.5f}")
                    pbar.update()
                    
                    # Generate test image every 10 epochs
                    if epoch % 10 == 0 and batch_idx == 0:
                        with torch.no_grad():
                            test_img = render_test_image(
                                model, poses[0], K, dataset.H, dataset.W, near, far
                            )
                            test_img_path = os.path.join(
                                output_dir, f"test_render_epoch_{epoch}.png"
                            )
                            visualise.create_comparison_visualization(
                                f"Epoch {epoch}",
                                images[0],
                                "Ground Truth",
                                test_img,
                                "Rendered",
                                test_img_path,
                            )
                    
                    # Early stopping for demo
                    if batch_idx >= 10:  # Process 10 batches per epoch for demo
                        break
            
            # Log epoch statistics
            avg_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch+1}/{n_epochs}, Loss: {avg_loss:.5f}")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = os.path.join(output_dir, f"nerf_model_epoch_{epoch+1}.pt")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                }, checkpoint_path)
                logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    # Save final model
    final_model_path = os.path.join(output_dir, "nerf_model_final.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_model_path)
    logger.info(f"Final model saved to {final_model_path}")
    
    # Render final test images
    render_test_views(model, poses, K, images, output_dir)
    
    # Create interactive visualization
    create_interactive_viewer(model, poses, K, output_dir)
    
    return model


def render_test_image(
    model: nn.Module, 
    pose: np.ndarray, 
    K: np.ndarray, 
    H: int, 
    W: int,
    near: float = 2.0,
    far: float = 6.0,
    n_samples: int = 64,
) -> np.ndarray:
    """Render a test image from a given camera pose.
    
    Args:
        model: NeRF model
        pose: Camera pose
        K: Intrinsic matrix
        H: Image height
        W: Image width
        near: Near plane distance
        far: Far plane distance
        n_samples: Number of samples per ray
    
    Returns:
        Rendered RGB image
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    
    # Create ray generator
    ray_gen = RayGenerator(H, W, K)
    
    # Generate rays
    origins, directions = ray_gen.get_rays(pose)
    
    # Render in chunks to avoid memory issues
    chunk_size = 4096
    n_chunks = (origins.shape[0] + chunk_size - 1) // chunk_size
    
    rgb_chunks = []
    for i in range(n_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, origins.shape[0])
        
        # Render chunk
        with torch.no_grad():
            rgb_chunk, _ = render_rays(
                model,
                origins[start:end],
                directions[start:end],
                near=near,
                far=far,
                n_samples=n_samples,
                perturb=False,
            )
        
        rgb_chunks.append(rgb_chunk.cpu())
    
    # Combine chunks
    rgb = torch.cat(rgb_chunks, dim=0)
    
    # Reshape to image
    rgb_img = rgb.reshape(H, W, 3).numpy()
    
    # Convert to uint8
    rgb_img = (rgb_img * 255).astype(np.uint8)
    
    return rgb_img


def render_test_views(
    model: nn.Module, 
    poses: List[np.ndarray], 
    K: np.ndarray, 
    images: List[np.ndarray],
    output_dir: str,
) -> None:
    """Render test views and compare to ground truth.
    
    Args:
        model: NeRF model
        poses: List of camera poses
        K: Intrinsic matrix
        images: List of ground truth images
        output_dir: Output directory
    """
    # Create renders directory
    renders_dir = os.path.join(output_dir, "renders")
    os.makedirs(renders_dir, exist_ok=True)
    
    # Get image dimensions
    H, W = images[0].shape[:2]
    
    # Render test views
    for i, pose in enumerate(tqdm(poses, desc="Rendering test views")):
        # Render image
        rendered = render_test_image(model, pose, K, H, W)
        
        # Save rendered image
        rendered_path = os.path.join(renders_dir, f"render_{i:03d}.png")
        import cv2
        cv2.imwrite(rendered_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
        
        # Create comparison visualization
        compare_path = os.path.join(renders_dir, f"compare_{i:03d}.png")
        visualise.create_comparison_visualization(
            f"View {i}",
            images[i],
            "Ground Truth",
            rendered,
            "NeRF Rendering",
            compare_path,
        )
    
    logger.info(f"Test views rendered to {renders_dir}")


def create_interactive_viewer(
    model: nn.Module, 
    poses: List[np.ndarray], 
    K: np.ndarray, 
    output_dir: str,
) -> None:
    """Create an interactive viewer for the NeRF model.
    
    Args:
        model: NeRF model
        poses: List of camera poses
        K: Intrinsic matrix
        output_dir: Output directory
    """
    logger.info("Creating interactive viewer (visualization)")
    
    # Get camera centers
    camera_centers = []
    for pose in poses:
        R = pose[:3, :3]
        t = pose[:3, 3]
        center = -R.T @ t
        camera_centers.append(center)
    
    camera_centers = np.array(camera_centers)
    
    # Calculate camera path for 360° orbit
    center = np.mean(camera_centers, axis=0)
    radius = 1.5 * np.max(np.linalg.norm(camera_centers - center, axis=1))
    
    # Generate camera poses for orbit
    n_frames = 30
    orbit_poses = []
    
    for i in range(n_frames):
        angle = 2 * np.pi * i / n_frames
        x = center[0] + radius * np.sin(angle)
        y = center[1]
        z = center[2] + radius * np.cos(angle)
        
        # Create camera pose matrix (looking at center)
        pos = np.array([x, y, z])
        forward = center - pos
        forward = forward / np.linalg.norm(forward)
        
        # Create coordinate system
        right = np.cross(np.array([0, 1, 0]), forward)
        right = right / np.linalg.norm(right)
        up = np.cross(forward, right)
        
        # Create rotation matrix
        R = np.stack([right, up, forward], axis=1)
        t = pos
        
        pose = np.zeros((3, 4))
        pose[:3, :3] = R
        pose[:3, 3] = t
        
        orbit_poses.append(pose)
    
    # Render orbit frames
    frames_dir = os.path.join(output_dir, "orbit")
    os.makedirs(frames_dir, exist_ok=True)
    
    # Use first image dimensions
    H, W = poses[0].shape[:2]
    H = 480  # Lower resolution for faster rendering
    W = 640
    
    # Adjust K for new resolution
    scale_x = W / poses[0].shape[1]
    scale_y = H / poses[0].shape[0]
    K_scaled = K.copy()
    K_scaled[0, 0] *= scale_x
    K_scaled[1, 1] *= scale_y
    K_scaled[0, 2] *= scale_x
    K_scaled[1, 2] *= scale_y
    
    # Render orbit frames
    for i, pose in enumerate(tqdm(orbit_poses, desc="Rendering orbit")):
        # Render image
        rendered = render_test_image(model, pose, K_scaled, H, W)
        
        # Save rendered image
        frame_path = os.path.join(frames_dir, f"frame_{i:03d}.png")
        import cv2
        cv2.imwrite(frame_path, cv2.cvtColor(rendered, cv2.COLOR_RGB2BGR))
    
    # Create GIF
    try:
        import imageio
        import glob
        
        frames = []
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
        
        for frame_file in frame_files:
            frames.append(imageio.imread(frame_file))
        
        gif_path = os.path.join(output_dir, "nerf_orbit.gif")
        imageio.mimsave(gif_path, frames, fps=10)
        
        logger.info(f"Orbit GIF saved to {gif_path}")
    except ImportError:
        logger.warning("imageio not available, skipping GIF creation")


def load_results(results_dir: str) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """Load reconstruction results for NeRF training.
    
    Args:
        results_dir: Path to reconstruction results directory
    
    Returns:
        Tuple of (images, poses, intrinsics)
    """
    logger.info(f"Loading reconstruction results from {results_dir}")
    
    # Load camera poses
    camera_file = os.path.join(results_dir, "cameras.npy")
    if not os.path.exists(camera_file):
        raise FileNotFoundError(f"Camera file not found: {camera_file}")
    
    with open(camera_file, "rb") as f:
        poses = np.load(f, allow_pickle=True)
    
    # Load intrinsics
    K_file = os.path.join(results_dir, "intrinsics.npy")
    if not os.path.exists(K_file):
        raise FileNotFoundError(f"Intrinsics file not found: {K_file}")
    
    with open(K_file, "rb") as f:
        K = np.load(f)
    
    # Load original images based on config
    config_file = os.path.join(results_dir, "../config.yaml")
    if not os.path.exists(config_file):
        config_file = Path(__file__).resolve().parent.parent / "config.yaml"
    
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    
    # Get input directory
    input_dir = config["io"]["input_dir"]
    
    # Read images
    images = []
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
        image_files.extend(list(Path(input_dir).glob(ext)))
        image_files.extend(list(Path(input_dir).glob(ext.upper())))
    
    image_files.sort()
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {input_dir}")
    
    # Ensure we have the same number of images as poses
    if len(image_files) < len(poses):
        logger.warning(
            f"Found {len(image_files)} images but {len(poses)} poses. "
            f"Using only the first {len(image_files)} poses."
        )
        poses = poses[:len(image_files)]
    elif len(image_files) > len(poses):
        logger.warning(
            f"Found {len(image_files)} images but only {len(poses)} poses. "
            f"Using only the first {len(poses)} images."
        )
        image_files = image_files[:len(poses)]
    
    # Read images
    for image_file in tqdm(image_files, desc="Reading images"):
        image = cv2.imread(str(image_file))
        if image is None:
            logger.warning(f"Failed to read {image_file}")
            continue
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)
    
    logger.info(f"Loaded {len(images)} images, {len(poses)} poses")
    
    return images, poses, K


def main():
    """Main function to parse arguments and run the NeRF training."""
    parser = argparse.ArgumentParser(description="NeRF Training")
    parser.add_argument(
        "--results", "-r", dest="results_dir", required=True,
        help="Path to reconstruction results directory"
    )
    parser.add_argument(
        "--output", "-o", dest="output_dir", default="results/nerf",
        help="Path to output directory"
    )
    parser.add_argument(
        "--config", "-c", dest="config_path", default=None,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--epochs", "-e", dest="epochs", type=int, default=10,
        help="Number of training epochs"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config_path:
        with open(args.config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config_path = Path(__file__).resolve().parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)["dense"]["nerf"]
    
    # Update config with command-line arguments
    config["n_epochs"] = args.epochs
    
    try:
        # Load reconstruction results
        images, poses, K = load_results(args.results_dir)
        
        # Train NeRF
        train_nerf(images, poses, K, config, args.output_dir)
    except Exception as e:
        logger.exception(f"Error running NeRF training: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
