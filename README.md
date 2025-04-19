# 3D Reconstruction using Linear Algebra

A Python implementation of Structure from Motion and Multi-View Stereo algorithms to convert a folder of 2D photos into an interactive 3D textured mesh, with every linear algebra step exposed in clean, well-documented code.

![3D Reconstruction Demo](docs/figures/reconstruction.gif)

*Note: The above animation will be generated when you run the pipeline with your own images.*

## Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/3d-recon-linear-algebra.git
cd 3d-recon-linear-algebra

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the pipeline on sample images
python scripts/run_pipeline.py --images data/sample --visualise
```

## Features

- **Feature Detection & Matching**: SIFT/ORB keypoint detection with FLANN matching
- **Geometric Computations**: Fundamental/Essential matrix estimation, triangulation, PnP
- **Bundle Adjustment**: Sparse optimization using Schur complement
- **Dense Reconstruction**: Patch-Match MVS or optional minimal NeRF implementation
- **Meshing & Texturing**: Poisson surface reconstruction with view-based texturing
- **Visualization**: 3D reconstruction viewer and diagnostic tools

## Where Linear Algebra Shows Up

This project extensively uses linear algebra at every stage of the 3D reconstruction pipeline:

- **Hartley Normalization**: Transforms points using translation and scaling matrices for better numerical stability
- **Fundamental Matrix Estimation**: Solves a homogeneous linear system using SVD to find the 8-point solution
- **Essential Matrix Decomposition**: Uses SVD to extract rotation and translation from the essential matrix
- **Triangulation**: Linear least squares solution via SVD to reconstruct 3D points from multiple views
- **Bundle Adjustment**: Builds and solves the normal equations (JᵀJ)δ = Jᵀr using Schur complement
- **Camera Projection**: Uses homogeneous coordinates and projection matrices to map 3D points to 2D
- **Pose Estimation**: SVD-based solutions for the PnP (Perspective-n-Point) problem
- **Dense MVS**: Uses matrix operations for stereo matching and depth map computation
- **Surface Reconstruction**: Linear systems in the Poisson reconstruction process
- **NeRF**: Linear operations in positional encoding and MLP layers

## Project Structure

```
3d-recon-linear-algebra/
├─ README.md
├─ LICENSE
├─ requirements.txt
├─ config.yaml
├─ data/
│   └─ sample/         # Demo photos (add your own here)
├─ results/            # Auto-generated reconstruction results
├─ docs/
│   └─ figures/        # Documentation images
├─ src/
│   ├─ feature.py      # Feature detection and matching
│   ├─ geometry.py     # Geometric computations (F, E, triangulation)
│   ├─ bundle.py       # Bundle adjustment optimization
│   ├─ dense.py        # Dense reconstruction (MVS/NeRF)
│   ├─ mesh.py         # Mesh reconstruction and texturing
│   ├─ evaluate.py     # Evaluation metrics and timing
│   └─ visualise.py    # Visualization utilities
├─ scripts/
│   ├─ run_pipeline.py # Main reconstruction pipeline
│   └─ train_nerf.py   # Optional NeRF training script
└─ tests/              # Unit and integration tests
```

## Adding Your Own Images

To run the reconstruction on your own images:

1. Add your images to a new folder (e.g., `data/my_photos/`)
2. Run the pipeline pointing to your images:

```bash
python scripts/run_pipeline.py --images data/my_photos --output results/my_reconstruction --visualise
```

For best results:
- Use 10-50 images with good overlap between consecutive views
- Ensure the subject is well-textured and static
- Avoid reflective or transparent surfaces
- Maintain consistent lighting across images

## Advanced Usage

### Configuration

You can customize the reconstruction parameters by editing `config.yaml` or creating your own:

```bash
python scripts/run_pipeline.py --images data/my_photos --config my_config.yaml
```

### NeRF Training

To train a Neural Radiance Field on your reconstruction:

```bash
python scripts/train_nerf.py --results results/my_reconstruction --output results/my_nerf --epochs 20
```

## Citation & License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

If you use this code in your research, please cite:

```
@software{3d-recon-linear-algebra,
  author = {Your Name},
  title = {3D Reconstruction using Linear Algebra},
  year = {2023},
  url = {https://github.com/yourusername/3d-recon-linear-algebra}
}
```

## Acknowledgments

This project builds upon key algorithms and approaches from the following papers:
- Hartley & Zisserman, "Multiple View Geometry in Computer Vision"
- Schönberger & Frahm, "Structure-from-Motion Revisited"
- Kazhdan et al., "Poisson Surface Reconstruction"
- Mildenhall et al., "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
