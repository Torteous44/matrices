"""Tests for the complete 3D reconstruction pipeline.

This module tests the entire pipeline, running from image input
to 3D reconstruction and ensuring the results satisfy quality metrics.
"""

import os
import sys
import unittest
from pathlib import Path
import shutil
import tempfile

import cv2
import numpy as np
import pytest

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src import feature, geometry, bundle, dense, mesh, evaluate
from scripts import run_pipeline


class TestPipeline(unittest.TestCase):
    """Test the full reconstruction pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures for pipeline tests."""
        # Create a temporary directory for test outputs
        cls.test_output_dir = tempfile.mkdtemp()
        
        # Path to sample data
        cls.sample_dir = Path(__file__).resolve().parent.parent / "data" / "sample"
        
        # Check if sample data exists, skip tests if not
        if not cls.sample_dir.exists() or len(list(cls.sample_dir.glob("*.jpg"))) == 0:
            cls.has_sample_data = False
            print("Warning: Sample data not found. Pipeline tests will be skipped.")
            print(f"Expected sample data at: {cls.sample_dir}")
        else:
            cls.has_sample_data = True
    
    @classmethod
    def tearDownClass(cls):
        """Clean up test fixtures."""
        # Remove the temporary directory
        if hasattr(cls, 'test_output_dir') and os.path.exists(cls.test_output_dir):
            shutil.rmtree(cls.test_output_dir)
    
    @pytest.mark.skipif("not TestPipeline.has_sample_data")
    def test_simple_pipeline(self):
        """Test a simplified version of the pipeline with minimal processing."""
        # Skip if no sample data
        if not self.has_sample_data:
            self.skipTest("Sample data not available")
        
        # Create test output subdirectory
        output_dir = os.path.join(self.test_output_dir, "simple_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run simplified pipeline
        metrics = run_simplified_pipeline(
            str(self.sample_dir),
            output_dir,
            max_images=5,  # Limit to first 5 images for speed
            feature_method="sift",
            skip_dense=True,  # Skip dense reconstruction for speed
            skip_mesh=True,   # Skip mesh reconstruction for speed
        )
        
        # Check that pipeline produced valid outputs
        points3d_path = os.path.join(output_dir, "points3d.npy")
        self.assertTrue(os.path.exists(points3d_path), "Pipeline should produce points3d.npy")
        
        # Load sparse points and check
        points3d = np.load(points3d_path)
        self.assertGreater(len(points3d), 100, "Should have reconstructed at least 100 points")
        
        # Check reprojection RMSE
        self.assertLess(metrics["rmse_reproj_px"], 2.0, "Reprojection RMSE should be less than 2.0 pixels")
    
    @pytest.mark.skipif("not TestPipeline.has_sample_data")
    def test_complete_pipeline(self):
        """Test the complete pipeline including dense reconstruction and mesh."""
        # Skip if no sample data
        if not self.has_sample_data:
            self.skipTest("Sample data not available")
        
        # Skip for CI environments or when running all tests to save time
        if 'CI' in os.environ or 'GITHUB_ACTIONS' in os.environ:
            self.skipTest("Skipping complete pipeline test in CI environment")
        
        # Create test output subdirectory
        output_dir = os.path.join(self.test_output_dir, "complete_test")
        os.makedirs(output_dir, exist_ok=True)
        
        # Run complete pipeline
        metrics = run_simplified_pipeline(
            str(self.sample_dir),
            output_dir,
            max_images=5,  # Limit to first 5 images for speed
            feature_method="sift",
            skip_dense=False,
            skip_mesh=False,
        )
        
        # Check that pipeline produced valid outputs
        mesh_path = os.path.join(output_dir, "mesh.obj")
        self.assertTrue(os.path.exists(mesh_path), "Pipeline should produce mesh.obj")
        
        # Check metrics
        self.assertLess(metrics["rmse_reproj_px"], 2.0, "Reprojection RMSE should be less than 2.0 pixels")
        self.assertGreater(metrics["dense_points"], 1000, "Should have reconstructed at least 1000 dense points")


def run_simplified_pipeline(
    image_dir: str,
    output_dir: str,
    max_images: int = 5,
    feature_method: str = "sift",
    skip_dense: bool = True,
    skip_mesh: bool = True,
) -> dict:
    """Run a simplified version of the reconstruction pipeline for testing.
    
    Args:
        image_dir: Path to directory containing images
        output_dir: Path to output directory
        max_images: Maximum number of images to process
        feature_method: Feature detection method (sift, orb)
        skip_dense: Whether to skip dense reconstruction
        skip_mesh: Whether to skip mesh reconstruction
    
    Returns:
        Dictionary of metrics from the reconstruction
    """
    # Create timer
    pipeline_timer = evaluate.Timer("Test Pipeline")
    pipeline_timer.start()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics
    metrics = evaluate.ReconstructionMetrics()
    
    # Stage 1: Read Images
    with evaluate.Timer("Read Images") as timer:
        # Read a limited number of images
        images = []
        image_files = sorted(list(Path(image_dir).glob("*.jpg")))[:max_images]
        
        for image_file in image_files:
            image = cv2.imread(str(image_file))
            if image is not None:
                images.append(image)
        
        metrics.update_stage_timing("read_images", timer.elapsed)
    
    print(f"Read {len(images)} images")
    
    # Estimate camera intrinsics
    K = run_pipeline.estimate_camera_matrix(images[0].shape[:2])
    
    # Stage 2: Detect Features and Match
    with evaluate.Timer("Feature Detection") as timer:
        matches = feature.detect_and_match(
            images, method=feature_method, ratio=0.75
        )
        metrics.update_stage_timing("detect_and_match", timer.elapsed)
    
    print(f"Found matches between {len(matches)} image pairs")
    
    # Stage 3: Build View Graph
    with evaluate.Timer("Build View Graph") as timer:
        adjacency = run_pipeline.build_view_graph(matches, min_matches=30)
        metrics.update_stage_timing("build_view_graph", timer.elapsed)
    
    # Stage 4: Incremental SfM
    with evaluate.Timer("Incremental SfM") as timer:
        # Create a simplified config
        config = {
            "sfm": {
                "min_inliers_initialization": 30,
                "min_inliers_registration": 20,
                "global_bundle_freq": 2,
                "global_bundle_max_rmse": 2.0,
                "init_pair_method": "max_inliers"
            }
        }
        
        poses, points3d, observations = run_pipeline.incremental_sfm(
            images, matches, K, config
        )
        metrics.update_stage_timing("incremental_sfm", timer.elapsed)
    
    print(f"Reconstructed {len(points3d)} points from {len(poses)} images")
    
    # Create colored point cloud
    colored_points = run_pipeline.create_point_cloud_with_colors(
        points3d, observations, images, poses, K
    )
    
    # Compute sparse reconstruction metrics
    metrics.compute_sparse_metrics(poses, points3d, observations, K)
    
    # Save results
    run_pipeline.save_results(
        output_dir, poses, points3d, observations, K
    )
    
    # Stage 5: Dense Reconstruction (optional)
    dense_cloud = None
    if not skip_dense:
        with evaluate.Timer("Dense Reconstruction") as timer:
            config = {"mvs": {"num_views": 2, "window_size": 5}}
            dense_cloud = dense.run_dense(
                images, poses, K, mode="mvs", config=config
            )
            metrics.update_stage_timing("dense_reconstruction", timer.elapsed)
        
        # Compute dense reconstruction metrics
        metrics.compute_dense_metrics(dense_cloud)
        
        # Save dense cloud
        dense_path = os.path.join(output_dir, "dense.npy")
        with open(dense_path, "wb") as f:
            np.save(f, dense_cloud)
    
    # Stage 6: Mesh Reconstruction (optional)
    mesh_obj = None
    if not skip_mesh and dense_cloud is not None:
        with evaluate.Timer("Mesh Reconstruction") as timer:
            config = {"poisson": {"depth": 6}}
            mesh_obj = mesh.create_textured_mesh(
                dense_cloud, images, poses, K, config
            )
            metrics.update_stage_timing("mesh_reconstruction", timer.elapsed)
        
        # Save mesh
        mesh_path = os.path.join(output_dir, "mesh.obj")
        mesh.save_mesh(mesh_obj, mesh_path, "obj")
    
    # Update final metrics
    metrics.update("runtime_s", pipeline_timer.elapsed)
    
    # Save metrics
    metrics_dict = metrics.to_dict()
    metrics_path = os.path.join(output_dir, "report.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics_dict, f, indent=2)
    
    print(metrics.summary())
    
    return metrics_dict


if __name__ == "__main__":
    unittest.main()
