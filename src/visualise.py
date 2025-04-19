"""Visualization utilities for 3D reconstruction.

This module provides visualization functions for 3D reconstruction results,
including camera poses, sparse and dense point clouds, and meshes.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

logger = logging.getLogger(__name__)


def create_camera_frustum(
    pose: np.ndarray,
    K: np.ndarray,
    width: float = 1.0,
    height: float = 0.75,
    depth: float = 0.5,
    color: Tuple[float, float, float] = (0.8, 0.2, 0.8)
) -> o3d.geometry.LineSet:
    """Create a camera frustum for visualization.

    Args:
        pose: 3x4 camera pose matrix [R|t]
        K: 3x3 camera intrinsic matrix
        width: Width of the frustum base
        height: Height of the frustum base
        depth: Depth of the frustum
        color: RGB color for the frustum

    Returns:
        Open3D LineSet representing the camera frustum
    """
    # Extract camera center and orientation
    R = pose[:3, :3]
    t = pose[:3, 3]
    
    # Calculate camera center in world space
    C = -R.T @ t
    
    # Calculate camera axis directions in world space
    z_axis = R.T @ np.array([0, 0, 1])
    y_axis = R.T @ np.array([0, 1, 0])
    x_axis = R.T @ np.array([1, 0, 0])
    
    # Calculate frustum corners
    depth_scale = depth / K[0, 0]
    hw = width / 2
    hh = height / 2
    
    # Define frustum points in camera coordinate system
    frustum_pts_cam = np.array([
        [0, 0, 0],               # Camera center
        [hw, hh, depth],         # Top-right
        [-hw, hh, depth],        # Top-left
        [-hw, -hh, depth],       # Bottom-left
        [hw, -hh, depth]         # Bottom-right
    ])
    
    # Transform to world coordinate system
    frustum_pts_world = []
    for pt in frustum_pts_cam:
        if np.allclose(pt, [0, 0, 0]):
            # Camera center is already calculated
            pt_world = C
        else:
            # Frustum corners
            pt_cam = pt.reshape(3, 1)
            pt_world = C + pt[0] * x_axis + pt[1] * y_axis + pt[2] * z_axis
        
        frustum_pts_world.append(pt_world)
    
    # Define lines connecting frustum points
    lines = [
        [0, 1], [0, 2], [0, 3], [0, 4],  # Lines from center to corners
        [1, 2], [2, 3], [3, 4], [4, 1]   # Lines connecting corners
    ]
    
    colors = [color for _ in range(len(lines))]
    
    # Create LineSet object
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(frustum_pts_world)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    
    return line_set


def get_coordinate_frame(size: float = 1.0) -> o3d.geometry.TriangleMesh:
    """Create a coordinate frame visualization.

    Args:
        size: Size of the coordinate frame

    Returns:
        Open3D TriangleMesh representing the coordinate frame
    """
    return o3d.geometry.TriangleMesh.create_coordinate_frame(size=size)


def array_to_pcd(
    points: np.ndarray,
    colors: Optional[np.ndarray] = None
) -> o3d.geometry.PointCloud:
    """Convert numpy arrays to Open3D point cloud.

    Args:
        points: Nx3 array of point coordinates
        colors: Nx3 array of RGB colors (optional)

    Returns:
        Open3D PointCloud object
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # Normalize colors if needed
        if np.max(colors) > 1.0:
            colors = colors / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd


def show(
    poses: List[np.ndarray],
    K: Optional[np.ndarray] = None,
    sparse: Optional[np.ndarray] = None,
    dense: Optional[np.ndarray] = None,
    mesh: Optional[o3d.geometry.TriangleMesh] = None,
    save_path: Optional[str] = None,
    window_size: Tuple[int, int] = (1280, 720)
) -> None:
    """Visualize the reconstruction results.

    Args:
        poses: List of camera poses [R|t]
        K: Camera intrinsic matrix (required for camera frustum visualization)
        sparse: Sparse point cloud (optional)
        dense: Dense point cloud (optional)
        mesh: Triangle mesh (optional)
        save_path: Path to save screenshot (optional)
        window_size: Visualization window size
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_size[0], height=window_size[1])
    
    # Add coordinate frame
    frame = get_coordinate_frame()
    vis.add_geometry(frame)
    
    # Add camera frustums if K is provided
    if K is not None:
        for i, pose in enumerate(poses):
            # Use different colors for different cameras
            hue = float(i) / max(1, len(poses) - 1)
            color = plt.cm.viridis(hue)[:3]  # RGB color from colormap
            
            frustum = create_camera_frustum(pose, K, color=color)
            vis.add_geometry(frustum)
    
    # Add sparse point cloud if provided
    if sparse is not None and len(sparse) > 0:
        sparse_points = sparse[:, :3]
        
        # Check if colors are provided
        sparse_colors = None
        if sparse.shape[1] >= 6:
            sparse_colors = sparse[:, 3:6]
        
        sparse_pcd = array_to_pcd(sparse_points, sparse_colors)
        vis.add_geometry(sparse_pcd)
    
    # Add dense point cloud if provided
    if dense is not None and len(dense) > 0:
        dense_points = dense[:, :3]
        
        # Check if colors are provided
        dense_colors = None
        if dense.shape[1] >= 6:
            dense_colors = dense[:, 3:6]
        
        dense_pcd = array_to_pcd(dense_points, dense_colors)
        vis.add_geometry(dense_pcd)
    
    # Add mesh if provided
    if mesh is not None:
        vis.add_geometry(mesh)
    
    # Set view
    view_control = vis.get_view_control()
    view_control.set_zoom(0.5)
    
    # Optimize view
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 2.0
    
    # Update
    vis.poll_events()
    vis.update_renderer()
    
    # Save screenshot if path is provided
    if save_path is not None:
        vis.capture_screen_image(save_path)
        logger.info(f"Screenshot saved to {save_path}")
    
    # Run interactive view
    vis.run()
    vis.destroy_window()


def save_epipolar_visualization(
    img1: np.ndarray,
    img2: np.ndarray,
    F: np.ndarray,
    points1: np.ndarray,
    points2: np.ndarray,
    output_path: str,
    n_lines: int = 10
) -> None:
    """Create and save visualization of epipolar lines.

    Args:
        img1: First image
        img2: Second image
        F: Fundamental matrix from img1 to img2
        points1: Nx2 array of points in img1
        points2: Nx2 array of corresponding points in img2
        output_path: Path to save the visualization
        n_lines: Number of epipolar lines to draw
    """
    # Select a subset of points for visualization
    if len(points1) > n_lines:
        indices = np.random.choice(len(points1), n_lines, replace=False)
        vis_points1 = points1[indices]
        vis_points2 = points2[indices]
    else:
        vis_points1 = points1
        vis_points2 = points2
    
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display images
    axs[0].imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    axs[1].imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    
    # Get image dimensions
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    # Draw epipolar lines and points
    colors = plt.cm.tab10.colors
    
    for i, (pt1, pt2) in enumerate(zip(vis_points1, vis_points2)):
        color = colors[i % len(colors)]
        
        # Draw point in first image
        axs[0].plot(pt1[0], pt1[1], 'o', color=color, markersize=8)
        
        # Draw corresponding point in second image
        axs[1].plot(pt2[0], pt2[1], 'o', color=color, markersize=8)
        
        # Compute epipolar line in second image
        line = F @ np.array([pt1[0], pt1[1], 1])
        a, b, c = line
        
        # Draw epipolar line in second image
        if abs(b) > abs(a):
            # Line is more horizontal
            x1, x2 = 0, w2
            y1 = (-c - a * x1) / b
            y2 = (-c - a * x2) / b
        else:
            # Line is more vertical
            y1, y2 = 0, h2
            x1 = (-c - b * y1) / a
            x2 = (-c - b * y2) / a
        
        axs[1].plot([x1, x2], [y1, y2], '-', color=color, linewidth=1)
        
        # Compute epipolar line in first image
        line = F.T @ np.array([pt2[0], pt2[1], 1])
        a, b, c = line
        
        # Draw epipolar line in first image
        if abs(b) > abs(a):
            # Line is more horizontal
            x1, x2 = 0, w1
            y1 = (-c - a * x1) / b
            y2 = (-c - a * x2) / b
        else:
            # Line is more vertical
            y1, y2 = 0, h1
            x1 = (-c - b * y1) / a
            x2 = (-c - b * y2) / a
        
        axs[0].plot([x1, x2], [y1, y2], '-', color=color, linewidth=1)
    
    # Add titles
    axs[0].set_title("Image 1 with epipolar lines from Image 2")
    axs[1].set_title("Image 2 with epipolar lines from Image 1")
    
    # Remove axes
    for ax in axs:
        ax.axis('off')
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Epipolar visualization saved to {output_path}")


def create_reconstruction_animation(
    mesh: o3d.geometry.TriangleMesh,
    output_path: str,
    n_frames: int = 80,
    rotation_axis: Tuple[float, float, float] = (0, 1, 0)
) -> None:
    """Create an animated GIF showing the rotating reconstruction.

    Args:
        mesh: Open3D triangle mesh
        output_path: Path to save the animation GIF
        n_frames: Number of frames in the animation
        rotation_axis: Axis of rotation
    """
    try:
        import imageio
    except ImportError:
        logger.error("imageio is required for creating animations")
        return
    
    # Create temporary directory for frames
    import tempfile
    import os
    from pathlib import Path
    
    temp_dir = tempfile.mkdtemp()
    frames = []
    
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=800, height=600, visible=False)
    
    # Add coordinate frame
    frame = get_coordinate_frame()
    vis.add_geometry(frame)
    
    # Add mesh
    vis.add_geometry(mesh)
    
    # Set view
    view_control = vis.get_view_control()
    view_control.set_zoom(0.6)
    
    # Optimize view
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    
    # Normalize rotation axis
    axis = np.array(rotation_axis)
    axis = axis / np.linalg.norm(axis)
    
    # Render frames
    for i in range(n_frames):
        # Rotate mesh for smooth animation
        angle = 2 * np.pi * i / n_frames
        R = mesh.get_rotation_matrix_from_axis_angle(axis * angle)
        mesh_copy = o3d.geometry.TriangleMesh(mesh)
        mesh_copy.rotate(R, center=mesh_copy.get_center())
        
        # Update geometry
        vis.update_geometry(mesh_copy)
        
        # Update and capture
        vis.poll_events()
        vis.update_renderer()
        
        frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
        vis.capture_screen_image(frame_path)
        frames.append(imageio.imread(frame_path))
    
    # Create GIF
    imageio.mimsave(output_path, frames, fps=15)
    
    # Clean up
    vis.destroy_window()
    
    # Remove temporary files
    for i in range(n_frames):
        os.remove(os.path.join(temp_dir, f"frame_{i:04d}.png"))
    os.rmdir(temp_dir)
    
    logger.info(f"Animation saved to {output_path}")


def create_depth_map_visualization(
    image: np.ndarray,
    depth: np.ndarray,
    output_path: str,
    colormap: str = 'turbo'
) -> None:
    """Create visualization of a depth map.

    Args:
        image: RGB image
        depth: Depth map (same size as image)
        output_path: Path to save the visualization
        colormap: Colormap to use for depth visualization
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display image
    axs[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    axs[0].set_title("RGB Image")
    axs[0].axis('off')
    
    # Create depth mask (for invalid/missing depth)
    valid_mask = depth > 0
    
    # Normalize depth for display
    depth_normalized = np.zeros_like(depth)
    if np.any(valid_mask):
        min_depth = np.min(depth[valid_mask])
        max_depth = np.max(depth[valid_mask])
        
        # Normalize to [0, 1]
        if max_depth > min_depth:
            depth_normalized[valid_mask] = (depth[valid_mask] - min_depth) / (max_depth - min_depth)
    
    # Apply colormap
    colormap_fn = plt.get_cmap(colormap)
    depth_colored = colormap_fn(depth_normalized)
    
    # Set alpha to 0 for invalid depth
    depth_colored[~valid_mask, 3] = 0
    
    # Display depth
    axs[1].imshow(depth_colored)
    axs[1].set_title(f"Depth Map (min: {np.min(depth[valid_mask]):.2f}, max: {np.max(depth[valid_mask]):.2f})")
    axs[1].axis('off')
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap_fn)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=axs[1], fraction=0.046, pad=0.04)
    cbar.set_label("Depth")
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Depth map visualization saved to {output_path}")


def visualize_camera_graph(
    camera_positions: np.ndarray,
    adjacency_matrix: np.ndarray,
    output_path: str
) -> None:
    """Visualize the camera connectivity graph.

    Args:
        camera_positions: Nx3 array of camera positions
        adjacency_matrix: NxN binary adjacency matrix
        output_path: Path to save the visualization
    """
    # Create 3D figure
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot camera positions
    ax.scatter(
        camera_positions[:, 0],
        camera_positions[:, 1],
        camera_positions[:, 2],
        c='blue',
        marker='o',
        s=50,
        label='Cameras'
    )
    
    # Plot connections
    for i in range(len(camera_positions)):
        for j in range(i+1, len(camera_positions)):
            if adjacency_matrix[i, j]:
                ax.plot(
                    [camera_positions[i, 0], camera_positions[j, 0]],
                    [camera_positions[i, 1], camera_positions[j, 1]],
                    [camera_positions[i, 2], camera_positions[j, 2]],
                    'r-',
                    alpha=0.5,
                    linewidth=1
                )
    
    # Add labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Connectivity Graph')
    
    # Set aspect ratio
    ax.set_box_aspect([1, 1, 1])
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Camera graph visualization saved to {output_path}")


# Helper functions for batch visualization

def create_comparison_visualization(
    title: str,
    left_img: np.ndarray,
    left_title: str,
    right_img: np.ndarray,
    right_title: str,
    output_path: str
) -> None:
    """Create a side-by-side comparison visualization.

    Args:
        title: Overall title
        left_img: Left image
        left_title: Left image title
        right_img: Right image
        right_title: Right image title
        output_path: Path to save the visualization
    """
    # Create figure
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    
    # Display images
    axs[0].imshow(left_img)
    axs[0].set_title(left_title)
    axs[0].axis('off')
    
    axs[1].imshow(right_img)
    axs[1].set_title(right_title)
    axs[1].axis('off')
    
    # Add overall title
    fig.suptitle(title, fontsize=16)
    
    # Tight layout
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # Adjust for the overall title
    
    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Comparison visualization saved to {output_path}")


def create_reconstruction_summary_image(
    sparse_cloud: np.ndarray,
    dense_cloud: Optional[np.ndarray],
    mesh: Optional[o3d.geometry.TriangleMesh],
    poses: List[np.ndarray],
    K: np.ndarray,
    output_path: str
) -> None:
    """Create a summary image of the reconstruction.

    Args:
        sparse_cloud: Sparse point cloud
        dense_cloud: Dense point cloud (optional)
        mesh: Triangle mesh (optional)
        poses: List of camera poses
        K: Camera intrinsic matrix
        output_path: Path to save the visualization
    """
    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=720, visible=False)
    
    # Add coordinate frame
    frame = get_coordinate_frame()
    vis.add_geometry(frame)
    
    # Add camera frustums
    for i, pose in enumerate(poses):
        # Use different colors for different cameras
        hue = float(i) / max(1, len(poses) - 1)
        color = plt.cm.viridis(hue)[:3]  # RGB color from colormap
        
        frustum = create_camera_frustum(pose, K, color=color)
        vis.add_geometry(frustum)
    
    # Add sparse point cloud
    if sparse_cloud is not None and len(sparse_cloud) > 0:
        sparse_points = sparse_cloud[:, :3]
        
        # Check if colors are provided
        sparse_colors = None
        if sparse_cloud.shape[1] >= 6:
            sparse_colors = sparse_cloud[:, 3:6]
        
        sparse_pcd = array_to_pcd(sparse_points, sparse_colors)
        vis.add_geometry(sparse_pcd)
    
    # Add dense point cloud
    if dense_cloud is not None and len(dense_cloud) > 0:
        dense_points = dense_cloud[:, :3]
        
        # Check if colors are provided
        dense_colors = None
        if dense_cloud.shape[1] >= 6:
            dense_colors = dense_cloud[:, 3:6]
        
        dense_pcd = array_to_pcd(dense_points, dense_colors)
        vis.add_geometry(dense_pcd)
    
    # Add mesh
    if mesh is not None:
        vis.add_geometry(mesh)
    
    # Set view
    view_control = vis.get_view_control()
    view_control.set_zoom(0.5)
    
    # Optimize view
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # Dark background
    opt.point_size = 2.0
    
    # Update
    vis.poll_events()
    vis.update_renderer()
    
    # Save screenshot
    vis.capture_screen_image(output_path)
    vis.destroy_window()
    
    logger.info(f"Reconstruction summary image saved to {output_path}")
