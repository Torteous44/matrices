"""Mesh reconstruction and texturing module.

This module implements mesh reconstruction from point clouds using
Poisson surface reconstruction and texture mapping.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import open3d as o3d
from tqdm import tqdm

logger = logging.getLogger(__name__)


def poisson_reconstruct(
    pcd: np.ndarray,
    depth: int = 8,
    scale: float = 1.1,
    samples_per_node: float = 1.5
) -> Union[o3d.geometry.TriangleMesh, None]:
    """Reconstruct mesh from point cloud using Poisson surface reconstruction.

    Args:
        pcd: Nx3 array of points or Nx6 array of points+normals
        depth: Maximum depth of the octree used for reconstruction
        scale: Scale factor for reconstruction
        samples_per_node: Minimum number of samples per octree node

    Returns:
        Reconstructed triangle mesh or None if reconstruction fails
    """
    start_time = time.perf_counter()
    logger.info(f"Starting Poisson reconstruction with depth={depth}")
    
    # Check input point cloud
    if pcd.shape[0] < 10:
        logger.error("Not enough points for mesh reconstruction")
        return None
    
    # Create Open3D point cloud
    o3d_pcd = o3d.geometry.PointCloud()
    
    # Add points
    if pcd.shape[1] >= 3:
        o3d_pcd.points = o3d.utility.Vector3dVector(pcd[:, :3])
    else:
        logger.error("Input point cloud must have at least 3 columns (XYZ)")
        return None
    
    # Add colors if available
    if pcd.shape[1] >= 6:
        # Normalize colors to [0, 1]
        colors = pcd[:, 3:6].copy()
        if np.max(colors) > 1.0:
            colors = colors / 255.0
        o3d_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Check for normals
    has_normals = False
    if pcd.shape[1] >= 9:
        # If point cloud has normals (XYZ + RGB + normals)
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd[:, 6:9])
        has_normals = True
    
    # Estimate normals if not provided
    if not has_normals:
        logger.info("Estimating point cloud normals")
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        o3d_pcd.orient_normals_consistent_tangent_plane(k=15)
    
    # Run Poisson surface reconstruction
    logger.info(f"Running Poisson reconstruction with {o3d_pcd.points.shape[0]} points")
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        o3d_pcd, depth=depth, scale=scale, linear_fit=True
    )
    
    # Remove low density vertices
    if len(densities) > 0:
        density_threshold = np.quantile(densities, 0.05)  # Remove bottom 5% of low density vertices
        logger.debug(f"Density threshold: {density_threshold}, min: {np.min(densities)}, max: {np.max(densities)}")
        mesh.remove_vertices_by_mask(densities < density_threshold)
    
    elapsed_time = time.perf_counter() - start_time
    n_vertices = len(mesh.vertices)
    n_triangles = len(mesh.triangles)
    
    logger.info(
        f"Poisson reconstruction complete: {n_vertices} vertices, {n_triangles} triangles "
        f"(elapsed time: {elapsed_time:.2f}s)"
    )
    
    return mesh


def texture_mesh(
    mesh: o3d.geometry.TriangleMesh,
    images: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray,
    method: str = "view_blend",
    resolution: int = 2048
) -> o3d.geometry.TriangleMesh:
    """Apply texture to a mesh using simple view-projection color blending.

    Args:
        mesh: Triangle mesh to texture
        images: List of input images
        poses: List of camera poses [R|t]
        K: Camera intrinsic matrix
        method: Texturing method, currently only "view_blend" is supported
        resolution: Resolution of the texture map

    Returns:
        Textured triangle mesh
    """
    start_time = time.perf_counter()
    logger.info(f"Starting mesh texturing using {method} method")
    
    # Make a copy of the mesh
    textured_mesh = o3d.geometry.TriangleMesh(mesh)
    
    # Get mesh vertices
    vertices = np.asarray(textured_mesh.vertices)
    triangles = np.asarray(textured_mesh.triangles)
    
    # Initialize vertex colors
    vertex_colors = np.zeros((len(vertices), 3))
    vertex_weights = np.zeros(len(vertices))
    
    # For each image, project vertices and assign colors
    for i, (image, pose) in enumerate(tqdm(zip(images, poses), total=len(images), desc="Texturing")):
        # Convert to RGB if needed
        if len(image.shape) == 2:
            # Convert grayscale to RGB
            image_rgb = np.stack([image, image, image], axis=2)
        else:
            image_rgb = image
        
        # Get image dimensions
        h, w = image_rgb.shape[:2]
        
        # Extract rotation and translation
        R = pose[:3, :3]
        t = pose[:3, 3]
        
        # Project vertices into this image
        proj_matrix = K @ np.hstack((R, t.reshape(3, 1)))
        
        # Apply projection
        vertices_homogeneous = np.hstack((vertices, np.ones((len(vertices), 1))))
        proj_vertices = proj_matrix @ vertices_homogeneous.T
        
        # Normalize homogeneous coordinates
        proj_vertices = proj_vertices / proj_vertices[2]
        
        # Get pixel coordinates
        pixel_coords = proj_vertices[:2].T
        
        # Check which vertices are visible (inside image bounds)
        visible = (
            (pixel_coords[:, 0] >= 0) &
            (pixel_coords[:, 0] < w) &
            (pixel_coords[:, 1] >= 0) &
            (pixel_coords[:, 1] < h)
        )
        
        # Compute view direction for each vertex
        view_dirs = vertices - (-R.T @ t)
        view_dirs_norm = np.linalg.norm(view_dirs, axis=1)
        
        # Normalize view directions
        view_dirs = view_dirs / view_dirs_norm.reshape(-1, 1)
        
        # Vertex normals
        vertex_normals = np.asarray(textured_mesh.vertex_normals)
        
        # Compute angle between view direction and vertex normal
        cos_angles = np.abs(np.sum(view_dirs * vertex_normals, axis=1))
        
        # Weight by angle (prefer views that are perpendicular to surface)
        angle_weights = cos_angles.copy()
        
        # Only use visible vertices
        valid_vertices = visible & (angle_weights > 0.1)
        
        if np.sum(valid_vertices) == 0:
            continue
        
        # Sample colors from image
        valid_coords = pixel_coords[valid_vertices].astype(np.int32)
        valid_colors = image_rgb[valid_coords[:, 1], valid_coords[:, 0]] / 255.0
        
        # Update vertex colors and weights
        vertex_colors[valid_vertices] += valid_colors * angle_weights[valid_vertices].reshape(-1, 1)
        vertex_weights[valid_vertices] += angle_weights[valid_vertices]
    
    # Normalize colors by weights
    valid_weights = vertex_weights > 0
    vertex_colors[valid_weights] /= vertex_weights[valid_weights].reshape(-1, 1)
    
    # Set default color for vertices with no weight
    vertex_colors[~valid_weights] = [0.7, 0.7, 0.7]  # Gray
    
    # Set vertex colors in mesh
    textured_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    
    elapsed_time = time.perf_counter() - start_time
    logger.info(
        f"Mesh texturing complete: {np.sum(valid_weights)}/{len(vertices)} vertices colored "
        f"(elapsed time: {elapsed_time:.2f}s)"
    )
    
    return textured_mesh


def create_textured_mesh(
    pcd: np.ndarray,
    images: List[np.ndarray],
    poses: List[np.ndarray],
    K: np.ndarray,
    config: Optional[Dict] = None
) -> o3d.geometry.TriangleMesh:
    """Create a textured mesh from a point cloud.

    Args:
        pcd: Point cloud as Nx6 array (XYZ + RGB)
        images: List of input images
        poses: List of camera poses [R|t]
        K: Camera intrinsic matrix
        config: Optional configuration parameters

    Returns:
        Textured triangle mesh
    """
    if config is None:
        config = {}
    
    # Get reconstruction parameters
    poisson_depth = config.get("poisson_depth", 8)
    poisson_scale = config.get("poisson_scale", 1.1)
    poisson_samples_per_node = config.get("poisson_samples_per_node", 1.5)
    texture_method = config.get("texture_method", "view_blend")
    texture_resolution = config.get("texture_resolution", 2048)
    
    # Create mesh
    mesh = poisson_reconstruct(
        pcd,
        depth=poisson_depth,
        scale=poisson_scale,
        samples_per_node=poisson_samples_per_node
    )
    
    if mesh is None:
        logger.error("Mesh reconstruction failed")
        # Return an empty mesh
        return o3d.geometry.TriangleMesh()
    
    # Apply texture
    textured_mesh = texture_mesh(
        mesh,
        images,
        poses,
        K,
        method=texture_method,
        resolution=texture_resolution
    )
    
    return textured_mesh


def save_mesh(
    mesh: o3d.geometry.TriangleMesh,
    output_path: str,
    file_format: str = "obj"
) -> bool:
    """Save mesh to file.

    Args:
        mesh: Triangle mesh to save
        output_path: Output file path
        file_format: Output file format (obj, ply, etc.)

    Returns:
        True if successful, False otherwise
    """
    # Ensure mesh has correct topology (needed for some file formats)
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.compute_vertex_normals()
    
    # Save mesh
    try:
        if file_format.lower() == "obj":
            o3d.io.write_triangle_mesh(output_path, mesh, write_vertex_colors=True)
        else:
            o3d.io.write_triangle_mesh(output_path, mesh)
        logger.info(f"Mesh saved to {output_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save mesh: {e}")
        return False


# Placeholder for additional mesh reconstruction methods (TODOs)

def marching_cubes() -> None:
    """TODO: Implement marching cubes algorithm for mesh reconstruction."""
    pass


def delaunay_triangulation() -> None:
    """TODO: Implement Delaunay triangulation for mesh reconstruction."""
    pass


def refined_texture_mapping() -> None:
    """TODO: Implement refined texture mapping with seam optimization."""
    pass
